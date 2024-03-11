## 这里是UPFWI的正演部分,负责将InversionNet生成的v_map使用有限差分方法获取seis_data
import numpy as np
import torch
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class Wave_Forward:
    def __init__(self,
                 velocity_model,
                 x_grid_nums,
                 z_grid_nums,
                 dx,
                 bgnums,
                 nt,
                 dt,
                 freq,
                 s_loc,
                 r_loc,
                 ):
        '''
        params:
            velocity_model: 速度图模型  array-like
            x_grid_nums   : x轴网格数量 int
            z_grid_nums   : z轴网格数量 int
            dx            : 网格的间隔(m) float/int dx = dz
            bgnums        : 吸收边界的网格数    int
            nt            : 时间步  int
            dt            : 采样时间间隔(s)  1/freq >> 2*dt(避免假频)   float
            freq          : ricker子波的中心频率(HZ) float/int
            s_loc         : source炮的位置(m)  tuple   (x,z)
            r_loc         : receiver的位置(m)   [(x1,z1),(x2,z2),...,(xn,zn)]
            record_gap    : 保存波场的时间步间隔(default:20)
            save_seis     : 用来保存地震图的文件名，默认在./output路径下
            save_wavefield: 用来保存波场图的文件名，默认在./output路径下
        '''
        self.velocity_model = velocity_model.to(device)
        self.x_grid_nums = x_grid_nums
        self.z_grid_nums = z_grid_nums
        self.dx = dx
        self.dz = dx
        self.bgnums = bgnums
        self.nt = nt
        self.dt = dt
        self.freq = freq
        self.s_loc = s_loc
        self.r_loc = r_loc
        self.wavefiled = None
        self.seis = None # 地震图
    def forward(self):
        '''
        正演函数
        return: 
            seis: 地震波图 array-like (nt,len(r_loc))
            wavefield:  波场快照
        '''
        ## init
        # create output folder
        # 接收器数量
        receiver_nums = len(self.r_loc)
        # 初始化地震图矩阵
        self.wavefiled = torch.zeros((self.nt,receiver_nums,receiver_nums)).to(device)
        # 拓展后的速度图
        pad_vel = self.pad_velocity().to(device)

        # 吸收边界条件
        abc = self.ABC_coef(pad_vel).to(device)
        kappa = abc*self.dt
        # ricker wavelet
        ricker = self.ricker_wavelet()
        # 炮与接收器的索引位置
        isx,isz,irx,irz = self._adjust_loc()
        # 差分系数 c_{-i}=c_{i}
        c0=-205.0/72.0;c1=8.0/5.0;c2=-1.0/5.0;c3=8.0/315.0;c4=-1.0/560.0
        a0 = -2.5;a1 = 4/3;a2 = -1/12
        # 初始化$p^{t}_{x,z}$与$p^{t-1}_{x,z}$
        p_t = torch.zeros_like(pad_vel).to(device)
        p_t_1 = torch.zeros_like(pad_vel).to(device)
        # 对应$\alpha=\left(\frac{\Delta t \cdot v}{\Delta x}\right)^2$
        alpha = (pad_vel*self.dt/self.dx)**2
        temp1 = (2-kappa)  # 不同差分格式,系数不一样
        temp2 = (1-kappa)
        beta_dt = (pad_vel*self.dt)**2
        ## start iterate
        laplace_pt = torch.zeros((3,self.nt,receiver_nums,receiver_nums)).to(device)
        for it in range(self.nt):
            ## eighth order
            # p = temp1*p_t-temp2*p_t_1+alpha*(
            #     2*c0*p_t+\
            #     c1*(torch.roll(p_t,1,dims=1)+torch.roll(p_t,-1,dims=1)+torch.roll(p_t,1,dims=0)+torch.roll(p_t,-1,dims=0))+\
            #     c2*(torch.roll(p_t,2,dims=1)+torch.roll(p_t,-2,dims=1)+torch.roll(p_t,2,dims=0)+torch.roll(p_t,-2,dims=0))+\
            #     c3*(torch.roll(p_t,3,dims=1)+torch.roll(p_t,-3,dims=1)+torch.roll(p_t,3,dims=0)+torch.roll(p_t,-3,dims=0))+\
            #     c4*(torch.roll(p_t,4,dims=1)+torch.roll(p_t,-4,dims=1)+torch.roll(p_t,4,dims=0)+torch.roll(p_t,-4,dims=0))
            # )

            # forth order
            p = temp1*p_t-temp2*p_t_1+alpha*(
                2*a0*p_t+\
                a1*(torch.roll(p_t,1,dims=1)+torch.roll(p_t,-1,dims=1)+torch.roll(p_t,1,dims=0)+torch.roll(p_t,-1,dims=0))+\
                a2*(torch.roll(p_t,2,dims=1)+torch.roll(p_t,-2,dims=1)+torch.roll(p_t,2,dims=0)+torch.roll(p_t,-2,dims=0))
            )


            p[isz,isx] = p[isz,isx] - beta_dt[isz,isx]*ricker[it]
            self.wavefiled[it] = p[self.bgnums:self.bgnums+receiver_nums,self.bgnums:self.bgnums+receiver_nums]
            # update
            p_t_1 = p_t
            p_t = p
        self.seis = self.wavefiled[:,0,:]
        return self.seis.to('cpu')

    def ABC_coef(self,pad_vel):
        '''
        吸收边界条件
        params:
            pad_vel:扩展后的速度图(利用pad_velocity函数产生)
        return:
            damp: 吸收边界  array-like  damp.shape=pad_vel.shape
                  在吸收边界上值离边界距离递增,在其他区域值为0
        '''
        nz_pad,nx_pad = pad_vel.shape
        vel_min = pad_vel.min()
        nz = nz_pad-2*self.bgnums
        nx = nx_pad-2*self.bgnums
        L = (self.bgnums-1)*self.dx
        k = 3.0*vel_min*math.log(1e7)/(2.0*L)
        damp1d = k*(torch.arange(self.bgnums).to(device)*self.dx/L)**2
        damp = torch.zeros((nz_pad,nx_pad))
        for iz in range(nz_pad):
            damp[iz,:self.bgnums] = torch.flip(damp1d,[0])
            damp[iz,nx+self.bgnums:nx+2*self.bgnums] = damp1d
        for ix in range(self.bgnums,self.bgnums+nx):
            damp[:self.bgnums,ix] = torch.flip(damp1d,[0])
            damp[self.bgnums+nz:2*self.bgnums+nz,ix] = damp1d
        return damp
    def pad_velocity(self):
        '''
        扩展速度图边界
        return:
            pad_vel 扩展边界后的速度图
            
        example:
            vel = [[1,2,3],
                   [4,5,6],
                   [7,8,9]]
            bgnums = 2
            
            ==>
            
            pad_vel = [[1, 1, 1, 2, 3, 3, 3],
                       [1, 1, 1, 2, 3, 3, 3],
                       [1, 1, 1, 2, 3, 3, 3],
                       [4, 4, 4, 5, 6, 6, 6],
                       [7, 7, 7, 8, 9, 9, 9],
                       [7, 7, 7, 8, 9, 9, 9],
                       [7, 7, 7, 8, 9, 9, 9]]
        '''
        v1 = torch.tile(self.velocity_model[:,0].reshape(-1,1),[1,self.bgnums])
        v2 = torch.tile(self.velocity_model[:,-1].reshape(-1,1),[1,self.bgnums])
        pad_vel = torch.concatenate((v1,self.velocity_model,v2),axis=1)
        v1 = torch.tile(pad_vel[0,:].reshape(1,-1),[self.bgnums,1])
        v2 = torch.tile(pad_vel[-1,:].reshape(1,-1),[self.bgnums,1])
        pad_vel = torch.concatenate((v1,pad_vel,v2),axis=0)
        return pad_vel
    
    def ricker_wavelet(self):
        '''
        产生ricker子波

        params:   
                freq:ricker子波的中心频率
                dt  :采样时间间隔 1/freq >> 2*dt(避免假频)
        '''
        # 2.2除以freq是为了完整的采样到波动段
        nt = 2.2/self.freq/self.dt
        nt = 2*math.floor(nt/2)+1
        period = math.floor(nt/2)
        k = torch.arange(1,nt+1)
        alpha = (period-k+1)*self.freq*self.dt*math.pi
        beta = alpha**2
        ricker = (1-beta*2)*torch.exp(-beta)
        if len(ricker) < self.nt:
            ricker = torch.concatenate([ricker,torch.zeros(self.nt-len(ricker))])
        return ricker
    def _adjust_loc(self):
        '''
        由于添加了吸收边界层，接收器与炮的绝对位置发生了变化
        return:
            isx:    炮在扩充边界后的x轴索引位置
            isz:    炮在扩充边界后的z轴索引位置
            irx:    接收器在扩充边界后的x轴索引位置
            irz:    接收器在扩充边界后的z轴索引位置
        '''
        isx = int(self.s_loc[0]/self.dx)+self.bgnums
        isz = int(self.s_loc[1]/self.dz)+self.bgnums
        irx = [int(self.r_loc[i][0]/self.dx)+self.bgnums for i in range(len(self.r_loc))]
        irz = [int(self.r_loc[i][1]/self.dz)+self.bgnums for i in range(len(self.r_loc))]
        return (isx,isz,irx,irz)

if __name__ == '__main__':
    v=torch.ones((70,70))*2000
    dt = 0.001
    dx = dz = 10
    nt = 1000
    s_loc = (350,0)
    pml = 100
    freq = 15
    model = Wave_Forward(velocity_model=v,
                        x_grid_nums=70,
                        z_grid_nums=70,
                        dx=dx,
                        bgnums=pml,
                        nt=nt,
                        dt=dt,
                        freq=freq,
                        s_loc=s_loc,
                        r_loc=[(i,0)for i in range(0,700,10)])
    seis = model.forward()
    print(seis[1])