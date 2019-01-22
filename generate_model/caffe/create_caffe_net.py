import sys
caffe_root = "/data/deeplearning/framework/caffe"
sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np

net_config = "./conv.prototxt"
model_file = "./conv.caffemodel"
caffe.set_mode_cpu()
net = caffe.Net(net_config, caffe.TEST)

#print("blobs {}\nparams {}".format(net.blobs.keys(), net.params.keys()))
print("================")
for param_name in net.params.keys():
    n = len(net.params[param_name])
    print ("-- %s  data number: %s" % (param_name, n))
    for i in range(n):
        shape = net.params[param_name][i].data.shape
        print ("%s data %s shape: %s" % (param_name, i, shape))
        net.params[param_name][i].data[...] = np.random.uniform(0, 1, shape)

net.save(model_file)
