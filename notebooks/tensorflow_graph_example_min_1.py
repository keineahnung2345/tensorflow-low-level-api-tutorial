import tensorflow as tf

g1 = tf.Graph()

with g1.as_default():
    my_input = tf.constant([-1,0,1], dtype=tf.float16, name="input")
    
    # Add a print operation in between our "input" operation and "A" operation
    my_printed_input = tf.Print(my_input, [], message="Running the graph.", name="print")

    a = tf.square(my_printed_input, name="A")
    b = tf.cos(a, name="B")
    c = tf.sin(a, name="C")   
    d = tf.add(b, c, name="D")
    e = tf.floor(b, name="E")
    f = tf.sqrt(d, name="F") 

sess = tf.Session(graph=g1);
print("A:{}".format(sess.run(g1.get_operation_by_name("A").outputs)))
print("B:{}".format(sess.run(g1.get_operation_by_name("B").outputs)))
print("C:{}".format(sess.run(g1.get_operation_by_name("C").outputs)))
print("D:{}".format(sess.run(g1.get_operation_by_name("D").outputs)))
print("E:{}".format(sess.run(g1.get_operation_by_name("E").outputs)))
print("F:{}".format(sess.run(g1.get_operation_by_name("F").outputs)))