#error analysis
#import tensorflow as tf
import numpy as np
import vgg_model as vgg
import matplotlib.pyplot as plt



a = ['../dance_data/validation/latin/heJXZdT-ne0_358_0046.jpg', '../dance_data/validation/latin/heJXZdT-ne0_358_0054.jpg', '../dance_data/validation/latin/heJXZdT-ne0_358_0062.jpg', '../dance_data/validation/latin/heJXZdT-ne0_358_0070.jpg', '../dance_data/validation/latin/heJXZdT-ne0_358_0078.jpg', '../dance_data/validation/latin/heJXZdT-ne0_358_0086.jpg', '../dance_data/validation/latin/heJXZdT-ne0_358_0094.jpg', '../dance_data/validation/latin/heJXZdT-ne0_358_0102.jpg', '../dance_data/validation/latin/heJXZdT-ne0_358_0110.jpg', '../dance_data/validation/latin/heJXZdT-ne0_358_0118.jpg', '../dance_data/validation/latin/heJXZdT-ne0_358_0126.jpg', '../dance_data/validation/latin/heJXZdT-ne0_358_0134.jpg', '../dance_data/validation/foxtrot/iOCZp-CmW5w_030_0038.jpg', '../dance_data/validation/foxtrot/iOCZp-CmW5w_030_0046.jpg', '../dance_data/validation/foxtrot/iOCZp-CmW5w_030_0054.jpg', '../dance_data/validation/foxtrot/iOCZp-CmW5w_030_0062.jpg', '../dance_data/validation/foxtrot/iOCZp-CmW5w_030_0070.jpg', '../dance_data/validation/foxtrot/iOCZp-CmW5w_030_0078.jpg', '../dance_data/validation/foxtrot/iOCZp-CmW5w_030_0086.jpg', '../dance_data/validation/foxtrot/iOCZp-CmW5w_030_0094.jpg', '../dance_data/validation/foxtrot/iOCZp-CmW5w_030_0102.jpg', '../dance_data/validation/foxtrot/iOCZp-CmW5w_030_0110.jpg', '../dance_data/validation/foxtrot/iOCZp-CmW5w_030_0118.jpg', '../dance_data/validation/foxtrot/iOCZp-CmW5w_030_0126.jpg']



scores_1 = [ -3.8464134,  -20.30352,     -4.3166513,   -5.219655, -1.7548882, -4.186255, -4.101478, -2.9727733, -0.90768874, -7.267065]
#prediction is tango
#actual is latin

scores_2 =[-14.397215,   -12.275844,    -9.531621,    -3.3765314,   -9.595953
,   -5.006969,   -20.010965,   -13.43656,     -5.991462,    -8.717364  ] #prediction is foxtrot

classes = ['ballet', 'break', 'flamenco', 'foxtrot', 'latin', 'quickstep', 'square', 'swing', 'tango', 'waltz']



def calculate_softmax(x):
    #array is 1D
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# softmax_score_first_example = calculate_softmax(scores_1)
# print("sum: ", np.sum(softmax_score_first_example))
# print("Softmax of latin misclassified as tango: ", softmax_score_first_example)

# softmax_score_second_example = calculate_softmax(scores_2)
# print("sum: ", np.sum(softmax_score_second_example))
# print("Softmax of foxtrot: ", softmax_score_second_example)

# y_pos = np.arange(len(softmax_score_first_example))

# plt.bar(y_pos, softmax_score_first_example, align = 'center', alpha = 0.5)
# plt.ylabel('Probability of Class')
# plt.xticks(y_pos, classes, rotation=45)
# plt.savefig("Example_1.png")



def plot_example_probability(scores, title):
    softmax = calculate_softmax(scores)
    y_pos = np.arange(len(softmax))
    plt.bar(y_pos, softmax, align = 'center', alpha = 0.5)
    plt.ylabel('Probability of Class')
    plt.xticks(y_pos, classes, rotation=45)
    plt.title(title)
    plt.savefig(title + ".png")
    plt.close()

plot_example_probability(scores_1, "Example 1")
plot_example_probability(scores_2, "Example 2")
# def compute_saliency_maps(X, y, model):
#     """
#     Compute a class saliency map using the model for images X and labels y.

#     Input:
#     - X: Input images, numpy array of shape (N, H, W, 3)
#     - y: Labels for X, numpy of shape (N,)
#     - model: A SqueezeNet model that will be used to compute the saliency map.

#     Returns:
#     - saliency: A numpy array of shape (N, H, W) giving the saliency maps for the
#     input images.
#     """
#     saliency = None
#     # Compute the score of the correct class for each example.
#     # This gives a Tensor with shape [N], the number of examples.
#     #
#     # Note: this is equivalent to scores[np.arange(N), y] we used in NumPy
#     # for computing vectorized losses.
#     correct_scores = tf.gather_nd(model.scores,
#                                   tf.stack((tf.range(X.shape[0]), model.labels), axis=1))
#     ###############################################################################
#     # TODO: Produce the saliency maps over a batch of images.                     #
#     #                                                                             #
#     # 1) Compute the “loss” using the correct scores tensor provided for you.     #
#     #    (We'll combine losses across a batch by summing)                         #
#     # 2) Use tf.gradients to compute the gradient of the loss with respect        #
#     #    to the image (accessible via model.image).                               #
#     # 3) Compute the actual value of the gradient by a call to sess.run().        #
#     #    You will need to feed in values for the placeholders model.image and     #
#     #    model.labels.                                                            #
#     # 4) Finally, process the returned gradient to compute the saliency map.      #
#     ###############################################################################
#     dX = tf.gradients(correct_scores, model.image)
#     saliency_compute = tf.squeeze(tf.reduce_max(tf.abs(dX), axis = 4))
#     temp = {
#         model.image:X,
#         model.labels:y
#     }
#     saliency = sess.run(saliency_compute, feed_dict = temp)
#     ##############################################################################
#     #                             END OF YOUR CODE                               #
#     ##############################################################################
#     return saliency


# def show_saliency_maps(X, y, mask):
#     mask = np.asarray(mask)
#     Xm = X[mask]
#     ym = y[mask]

#     saliency = compute_saliency_maps(Xm, ym, model)

#     for i in range(mask.size):
#         plt.subplot(2, mask.size, i + 1)
#         plt.imshow(deprocess_image(Xm[i]))
#         plt.axis('off')
#         plt.title(class_names[ym[i]])
#         plt.subplot(2, mask.size, mask.size + i + 1)
#         plt.title(mask[i])
#         plt.imshow(saliency[i], cmap=plt.cm.hot)
#         plt.axis('off')
#         plt.gcf().set_size_inches(10, 4)
#     plt.show()

# mask = np.arange(1)
# show_saliency_maps(X, y, mask)




