import tensorflow as tf


def sess(x):
    sess =tf.Session()
    op = sess.run(x)
    print(x)
    print(op)
    '''
    writer = tf.summary.FileWriter('./Day5LOG',sess.graph)
    writer.close()
    '''
    sess.close()
    


x = tf.zeros([2,3],tf.int32)

sess(x)
#-------------------------------------------


X = tf.constant([[1,2,3],[4,5,6]],tf.int32)
X1 = tf.zeros([2,3],tf.int32)

sess(X)
sess(X1)
#---------------------------------------------

Y = tf.ones([2,3])
sess(Y)
#---------------------------------------------

A = tf.constant([[1,2,3],[4,5,6]],tf.int32)
A1 = tf.ones([2,3],tf.int32)
sess(A)
sess(A1)
#---------------------------------------------

B = tf.ones([3,2],tf.int32)
B1 = B*5
sess(B)
sess(B1)
#---------------------------------------------

C = tf.constant([[1,3,5],[4,6,8]],tf.float32)
sess(C)
#---------------------------------------------

D = tf.constant([[4,4,4],[4,4,4]])
sess(D)
#---------------------------------------------

E = tf.linspace(5.0,10.0,50)
sess(E)

#-----------------------------------------------
import numpy as np

F = tf.random_normal([2, 3], mean=0, stddev=2)
sess(F)

#-----------------------------------------------

G = tf.random_uniform([3,2],minval=0,maxval=2)
sess(G)

#------------------------------------------------

H = tf.constant([[1,2],[3,4],[5,6]])

H1 = tf.random_shuffle(H)
sess(H1)

#-------------------------------------------------

I= tf.random_normal([10, 10, 3])
I1 = tf.random_crop(I,[5,5,3])

sess(I1)

#---------------------------------------------------

J = tf.constant([[-1,-2,-3],[0,1,2]],tf.int32)
J1 = tf.zeros([2,3],tf.int32)

J2 = tf.not_equal(J,J1)
sess(J2)

#--------------------------------------------------
#Mathematical_Operations

P =tf.constant([[1,2,3],[4,5,6]])

Q =tf.constant([[1,3,5],[4,6,8]])

R =tf.constant([[4,4,4],[4,4,4]])


ad = tf.add(P,Q)
sess(ad)

#---------------------------------------------------

sub = tf.subtract(P,Q)
sess(sub)
#---------------------------------------------------

mul = tf.multiply(P,Q)
sess(mul)

#--------------------------------------------------

mul1 = tf.multiply(P,5)
sess(mul1)
#--------------------------------------------------

ad3 = tf.add_n([P,Q,R])
sess(ad3)

#--------------------------------------------------

#variables & Placeholders

w = tf.Variable(1.0,name='weight')

init_op = tf.global_variables_initializer()
sess=tf.Session()
sess.run(init_op)
print(sess.run(w))
sess.close()


#--------------------------------------------------

p = tf.Variable(5)
q = tf.Variable(20)
r = tf.Variable(7)

addpqr = tf.Variable(p+q+r)
init_op = tf.global_variables_initializer()
sess=tf.Session()
sess.run(init_op)
print(sess.run(addpqr))
sess.close()

#---------------------------------------------------

z1 = tf.placeholder(tf.float32)


init_op = tf.global_variables_initializer()
sess=tf.Session()
sess.run(init_op)
print(sess.run(z1,feed_dict={z1:5}))
sess.close()

#-----------------------------------------------------

p1 = tf.placeholder(tf.float32)
p2 = tf.placeholder(tf.float32)
p3 = tf.placeholder(tf.float32)

psum = tf.add_n([p1,p2,p3])

init_op = tf.global_variables_initializer()
sess=tf.Session()
sess.run(init_op)
print(sess.run(psum,feed_dict={p1:5.0,p2:6.0,p3:8.0}))
sess.close()




