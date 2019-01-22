import tensorflow as tf

# 设置worker 地址与端口
worker_01 = "127.0.0.1:2222"
worker_02 = "127.0.0.1:2223"
worker_hosts = [worker_01, worker_02]
# Creates a `ClusterSpec`
cluster_spec = tf.train.ClusterSpec({"worker": worker_hosts})
# Creates a new server with ClusterSpec、job_name、task_index
server = tf.train.Server(cluster_spec, job_name="worker", task_index=0)

server.join()
