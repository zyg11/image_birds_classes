import numpy as np
from keras import backend as K
from keras.models import load_model
import matplotlib.pyplot as plt
from scipy.misc import imsave #以图像形式保存数组
img_width=150
img_height=150
def deprocess_image(x):
    #归一化张量:center on 0.,ensure std is 0.1
    x-=x.mean()
    x/=(x.std()+1e-5)
    x*=0.1

    #clip to[0,1]
    x+=0.5
    x=np.clip(x,0,1)

    # #转化成RGB
    x*=255
    if K.image_data_format()=='channels_first':
        x=x.transpose((1,2,0))
    x=np.clip(x,0,255).astype('uint8')
    return  x
model=load_model('E:/keras_data/data1/5class_model.h5')
print('model loaded')
model.summary()
input_img=model.layers[0].input
layer_name='conv1_1'
# filter_index=0

layer_dict = dict([(layer.name, layer) for layer in model.layers])
def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x/(K.sqrt(K.mean(K.square(x)))+1e-5)

for filter_index in range(31):#63
    #we only scan through the first 16 filters,but there are actually 512 of them
    print('Processing filter %d' % filter_index)
    # start_time=time.time()
    #we build a loss function that maximizes the activatio nof the nth filter of the layer considered
    layer_output=layer_dict[layer_name].output

    if K.image_data_format()=='channels_first':
        loss=K.mean(layer_output[:, filter_index, :,:])
    else:
        loss = K.mean(layer_output[:, :, :, filter_index])
    # we compute the gradient of the input picture wrt this loss
    grad=K.gradients(loss,input_img)[0]
    #归一化梯度
    grads=normalize(grad)
    # this function returns the loss and grads given the input picture
    iterate=K.function([input_img],[loss,grads])
    #开始梯度上升
    step=1.
    # we start from a gray image with some random noise
    input_img_data=np.random.random((1,img_width,img_height,3))
    input_img_data=(input_img_data-0.5)*20+128
    #梯度上升20步
    for i in range(20):
        loss_value,grads_value=iterate([input_img_data])#这里有问题
        input_img_data+=step*grads_value
        print('Current loss value:', loss_value)
        if loss_value<=0.:
            #某些滤波器卡住了，可以跳过
            break
    if loss_value > 0:
        img = deprocess_image(input_img_data[0])
# img=img.reshape(28,28)
img=np.array(img)
plt.imshow(img)
plt.show()

imsave('Birds_conv_1_1.png',img)