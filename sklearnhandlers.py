#!/usr/bin/python

from pymongo import MongoClient
import tornado.web

from tornado.web import HTTPError
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.options import define, options

from basehandler import BaseHandler

from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
import pickle
from bson.binary import Binary
import json
import numpy as np

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    #layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    #layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer

class PrintHandlers(BaseHandler):
    def get(self):
        '''Write out to screen the handlers used
        This is a nice debugging example!
        '''
        self.set_header("Content-Type", "application/json")
        self.write(self.application.handlers_string.replace('),','),\n'))

class TestTensorFlow(BaseHandler):
    def get(self):
        '''Test tensorflow
        '''
        mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
        # Parameters
        learning_rate = 0.001
        training_epochs = 15
        batch_size = 100
        display_step = 1

        # Network Parameters
        n_hidden_1 = 256 # 1st layer number of features
        #n_hidden_2 = 256 # 2nd layer number of features
        n_input = 784 # MNIST data input (img shape: 28*28)
        n_classes = 10 # MNIST total classes (0-9 digits)

        # tf Graph input
        x = tf.placeholder("float", [None, n_input])
        y = tf.placeholder("float", [None, n_classes])

        # Store layers weight & bias
        weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
           # 'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
        }
        biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1])),
            #'b2': tf.Variable(tf.random_normal([n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_classes]))
        }

        # Construct model
        pred = multilayer_perceptron(x, weights, biases)

        # Define loss and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        # Initializing the variables
        init = tf.initialize_all_variables()

        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)

            # Training cycle
            for epoch in range(training_epochs):
                avg_cost = 0.
                total_batch = int(mnist.train.num_examples/batch_size)
                # Loop over all batches
                for i in range(total_batch):
                    batch_x, batch_y = mnist.train.next_batch(batch_size)
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                                  y: batch_y})
                    # Compute average loss
                    avg_cost += c / total_batch
                # Display logs per epoch step
                if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch+1), "cost=", \
                        "{:.9f}".format(avg_cost))
            print("Optimization Finished!")

            # Test model
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

        self.write_json({"status": "ok"})

    def post(self):
        '''Save features
        '''
        data = json.loads(self.request.body.decode("utf-8"))
        print(data)
        self.write_json({"result": "ok"})

class UploadLabeledDatapointHandler(BaseHandler):
    def post(self):
        '''Save data point and class label to database
        '''
        data = json.loads(self.request.body.decode("utf-8"))
        sess  = data['dsid']

        for index, features in enumerate(data["features"]):      
            fvals = [float(val) for val in features]
            status = float(data["status"][index])
            dbid = self.db.labeledinstances.insert(
                {"feature":fvals,"status":status,"dsid":sess}
                );
        self.write_json({"result":"ok"})

class RequestNewDatasetId(BaseHandler):
    def get(self):
        '''Get a new dataset ID for building a new dataset
        '''
        a = self.db.labeledinstances.find_one(sort=[("dsid", -1)])
        newSessionId = float(a['dsid'])+1
        self.write_json({"dsid":newSessionId})

class UpdateModelForDatasetId(BaseHandler):
    def get(self):
        '''Train a new model (or update) for given dataset ID
        '''
        #dsid = self.get_int_arg("dsid",default=0)
        dsid = 201

        # create feature vectors from database
        # f=[]
        # lx=[]
        # ly=[]
        # for a in self.db.labeledinstances.find({"dsid":dsid}): 
        #     f.append([float(val) for val in a['feature']])
        #     lx.append(a['labelX'])
        #     ly.append(a['labelY'])

        # # fit the model to the data
        # # c1 = SVR(kernel='rbf')
        # # c2 = SVR(kernel='rbf')

        # c1 = linear_model.LinearRegression()
        # c2 = linear_model.LinearRegression()


        # if lx:
        #     c1.fit(f,lx) # training
        #     #lstar = c1.predict(f)
        #     #self.clf = c1
        #     #acc = sum(lstar==l)/float(len(l))
        #     bytes = pickle.dumps(c1)
        #     self.db.models.update({"dsid":98},
        #         {  "$set": {"model":Binary(bytes)}  },
        #         upsert=True)
        # if ly:
        #     c2.fit(f,ly) # training
        #     #lstar = c1.predict(f)
        #     #self.clf = c1
        #     #acc = sum(lstar==l)/float(len(l))
        #     bytes = pickle.dumps(c2)
        #     self.db.models.update({"dsid":99},
        #         {  "$set": {"model":Binary(bytes)}  },
        #         upsert=True)

        f=[]
        labels = []
        for a in self.db.labeledinstances.find({"dsid":dsid}): 
            f.append([float(val) for val in a['feature']])
            labels.append(a['status'])

        c3 = KNeighborsClassifier(n_neighbors=3);
        if labels:
            c3.fit(f, labels)
            bytes = pickle.dumps(c3)
            self.db.models.update({"dsid":202},
                {  "$set": {"model":Binary(bytes)}  },
                upsert=True)



        # send back the resubstitution accuracy
        # if training takes a while, we are blocking tornado!! No!!
        self.write_json({"resubAccuracy":"okkk"})

class PredictOneFromDatasetId(BaseHandler):
    def post(self):
        '''Predict the class of a sent feature vector
        '''
        data = json.loads(self.request.body.decode("utf-8"))    

        vals = data['features'];
        fvals = [float(val) for val in vals];
        fvals = np.array(fvals).reshape(1, -1)
        dsid  = data['dsid']

        # load the model from the database (using pickle)
        # we are blocking tornado!! no!!
        # print('Loading Model From DB')
        # print("Get feature")
        # print(fvals)
        # tmp = self.db.models.find_one({"dsid":98})
        # cl1 = pickle.loads(tmp['model'])
        # predLabelX = cl1.predict(fvals);
        # print("X:" + str(predLabelX))

        # tmp = self.db.models.find_one({"dsid":99})
        # cl2 = pickle.loads(tmp['model'])
        # predLabelY = cl2.predict(fvals);
        # print("Y:" + str(predLabelY))

        tmp = self.db.models.find_one({"dsid":202})
        cl3 = pickle.loads(tmp['model'])
        predLabel = cl3.predict(fvals);
        print("labels" + str(predLabel))
        y = predLabel[0]
        print("status: " + str(y))
        self.write_json({"status": y})
