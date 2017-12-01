import os,sys
import time
import numpy as np
from IPython import embed
import glob
from multiprocessing import Pool
import skvideo.io
import scipy.misc
import cv2 as cv
from PIL import Image

video_root='../data/ucf101/videos/'
op_cpp_tool='../TSN/lib/dense_flow/build/'
out_path='/local/MI/zqj/'

class TimeMaster(object):
    def __init__(self,):
        self.set_init()
    def set_init(self,):
        self.beg_time=0
        self.end_time=0
        self.n_times=0
        self.sum_time=0
    def start(self,):
        self.beg_time=time.time()
    def end(self,):
        self.end_time=time.time()
        self.sum_time+=self.end_time-self.beg_time
        self.n_times+=1
    def print_info(self,):
        info='totally cost time :{}'.format(self.sum_time)
        return info

def extract_op_tvl1(path):
    name=path.split('/')[-1].split('.')[0]+'_tvl1'
    op_fb_path=os.path.join(op_cpp_tool,'extract_gpu')
    out_frames_path=os.path.join(out_path,name)
    if not os.path.exists(out_frames_path):
        os.makedirs(out_frames_path)
    command='{} -f {} -x {} -y {} -i {} -b 20 -t 1 -d {} -s 1 -o {} -w {} -h {}'.format(
            op_fb_path,
            path,
            out_frames_path+'/flow_x',
            out_frames_path+'/flow_y',
            out_frames_path+'/',
            1,
            'dir',
            340,
            256)
    os.system(command)

def extract_op_fb(path):
    name=path.split('/')[-1].split('.')[0]+'_fb'
    op_tvl1_path=os.path.join(op_cpp_tool,'extract_cpu')
    out_frames_path=os.path.join(out_path,name)
    if not os.path.exists(out_frames_path):
        os.makedirs(out_frames_path)
    command='{} -f {} -x {} -y {} -i {} -b {} -o {}'.format(
        op_tvl1_path,
        path,
        out_frames_path+'/flow_x',
        out_frames_path+'/flow_y',
        out_frames_path+'/',
        20,
        'dir')
    os.system(command)

def ToImg(raw_flow,bound):
    flow=raw_flow
    flow[flow>bound]=bound
    flow[flow<-bound]=-bound
    flow-=-bound
    flow*=(256/float(2*bound))
    return flow

def save_flows(flows,image,save_dir,num,bound):
    #rescale to 0~255 with the bound setting
    flow_x=ToImg(flows[...,0],bound)
    flow_y=ToImg(flows[...,1],bound)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    #save the image
    save_img=os.path.join(save_dir,'img_{:05d}.jpg'.format(num))
    scipy.misc.imsave(save_img,image)
    #save the flows
    save_x=os.path.join(save_dir,'flow_x_{:05d}.jpg'.format(num))
    save_y=os.path.join(save_dir,'flow_y_{:05d}.jpg'.format(num))
    flow_x_img=Image.fromarray(flow_x)
    flow_y_img=Image.fromarray(flow_y)
    #embed()
    #flow_x_img.save(save_x)
    #flow_y_img.save(save_y)
    scipy.misc.imsave(save_x,flow_x_img)
    scipy.misc.imsave(save_y,flow_y_img)
    return 0

def python_extractor(op_method,videocapture,name,params):
    if videocapture.sum()==0:
        print 'could not initialize capturing {}'.format()
        exit()
    if op_method=='tvl1':
        calc_method=cv.createOptFlow_DualTVL1()
    else: #op_method=='fb'
        calc_method=cv.calcOpticalFlowFarneback

    out_frames_path=os.path.join(out_path,name)
    len_frame=len(videocapture)
    frame_num=0
    image,prev_image,gray,prev_gray=None,None,None,None
    num0=0
    while True:
        if num0>=len_frame:
            break
        frame=videocapture[num0]
        num0+=1
        if frame_num==0:
            image=np.zeros_like(frame)
            gray=np.zeros_like(frame)
            prev_gray=np.zeros_like(frame)
            prev_image=frame
            prev_gray=cv.cvtColor(prev_image,cv.COLOR_BGR2GRAY)
            frame_num+=1

            step_t=params['step']
            while step_t>1:
                num0+=1
                step_t-=1
            continue
        image=frame
        gray=cv.cvtColor(image,cv.COLOR_BGR2GRAY)

        frame_0=prev_gray
        frame_1=gray

        if op_method=='tvl1':
            OutFlow=calc_method.calc(frame_0,frame_1,None)
        else :
            OutFlow=calc_method(frame_0,frame_1,None,0.6,3,25,7,5,1.2,cv.OPTFLOW_FARNEBACK_GAUSSIAN)
        save_flows(OutFlow,image,out_frames_path,frame_num,params['bound'])
        prev_gray=gray
        prev_image=image
        frame_num+=1

        step_t=params['step']
        while step_t>1:
            num0+=1
            step_t-=1

def extract_op_fb_python(path):
    name=path.split('/')[-1].split('.')[0]+'_fb_python'
    op_method='fb'# alternatives between 'fb' and 'tvl1'
    videocapture=skvideo.io.vread(path)
    params={'step':1,'bound':20}
    python_extractor(op_method,videocapture,name,params)

def extract_op_tvl1_python(path):
    name=path.split('/')[-1].split('.')[0]+'_tvl1_python'
    op_method='tvl1'# alternatives between 'fb' and 'tvl1'
    videocapture=skvideo.io.vread(path)
    params={'step':1,'bound':20}
    python_extractor(op_method,videocapture,name,params)

def get_videos(video_root,num_videos):
    is_shuffle=0
    videos=glob.glob(os.path.join(video_root,'*','*.avi'))
    if not is_shuffle:
        videos.sort()
    return videos[:num_videos]

def main():

    num_videos=100
    file_list=get_videos(video_root,num_videos)

    fb_cpu_single=TimeMaster()
    tvl1_gpu_single=TimeMaster()
    fb_cpu_multi=TimeMaster()
    tvl1_gpu_multi=TimeMaster()
    fb_cpu_single_python=TimeMaster()
    tvl1_gpu_single_python=TimeMaster()
    fb_cpu_multi_python=TimeMaster()
    tvl1_gpu_multi_python=TimeMaster()

    '''-------------------------single core computing,cpp api--------------------------'''
    fb_cpu_single.start()
    print 'cpp api, single core, cpu, fb : {}\n'.format(fb_cpu_single.beg_time)
    for i,item in enumerate(file_list):
        extract_op_fb(item)
    fb_cpu_single.end()

    tvl1_gpu_single.start()
    print 'cpp api, single core, gpu, tvl1: {}\n'.format(tvl1_gpu_single.beg_time)
    for i,item in enumerate(file_list):
        extract_op_tvl1(item)
    tvl1_gpu_single.end()

    '''-------------------------multi cores computing,cpp api----------------------------'''
    multi_thread=8
    pool=Pool(multi_thread)
    fb_cpu_multi.start()
    print 'cpp api, multi cores, cpu, fb : {}\n'.format(fb_cpu_multi.beg_time)
    pool.map(extract_op_fb,file_list)
    fb_cpu_multi.end()
    tvl1_gpu_multi.start()
    print 'cpp api, multi cores, gpu, tvl1: {}\n'.format(tvl1_gpu_multi.beg_time)
    pool.map(extract_op_tvl1,file_list)
    tvl1_gpu_multi.end()

    '''------------------------single core computing, python api--------------------------'''
    fb_cpu_single_python.start()
    print 'python api,single core, cpu, fb : {}\n'.format(fb_cpu_single_python.beg_time)
    for i, item in enumerate(file_list):
        extract_op_fb_python(item)
    fb_cpu_single_python.end()

    tvl1_gpu_single_python.start()
    print 'python api,single core, gpu, tvl1: {}\n'.format(tvl1_gpu_single_python.beg_time)
    for i,item in enumerate(file_list):
        extract_op_tvl1_python(item)
    tvl1_gpu_single_python.end()

    '''------------------------multi cores computing, python api--------------------------'''
    multi_thread=8
    pool=Pool(multi_thread)
    fb_cpu_multi_python.start()
    print 'python api, multi cores, cpu, fb: {}\n'.format(fb_cpu_multi_python.beg_time)
    pool.map(extract_op_fb_python,file_list)
    fb_cpu_multi_python.end()

    tvl1_gpu_multi_python.start()
    print 'python api, multi cores tvl1, gpu, tvl1 :{}\n'.format(tvl1_gpu_multi_python.beg_time)
    pool.map(extract_op_tvl1_python,file_list)
    tvl1_gpu_multi_python.end()

    print 'all done at {}'.format(time.ctime())
    print 'op_fb_cpu cost {}s, average {}s per video\nop_tvl1_gpu cost {}s, average {}s per video\n'.\
    format(fb_cpu_single,fb_cpu_single/num_videos,tvl1_gpu_single,tvl1_gpu_single/num_videos)
    print 'op_multi_cpu cost {}s, average {}s per video\nop_multi_gpu cost {}s, average {}s per video\n'.\
    format(fb_cpu_multi,fb_cpu_multi/num_videos,tvl1_gpu_multi,tvl1_gpu_multi/num_videos)
    print 'op_fb_cpu_python cost {}s, average {}s per video\nop_tvl1_gpu_python cost {}s, average {}s per video\n'.\
    format(fb_cpu_single_python,fb_cpu_single_python/num_videos,tvl1_gpu_single_python,tvl1_gpu_single_python/num_videos)
    print 'op_multi_cpu_python cost {}s, average {}s per video\nop_multi_gpu_python cost {}s, average {}s per video\n'.\
    format(fb_cpu_multi_python,fb_cpu_multi_python/num_videos,tvl1_gpu_multi_python,tvl1_gpu_multi_python/num_videos)

if __name__=='__main__':
    main()
