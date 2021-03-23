import importlib
import sys
import tensorflow as tf
import numpy as np


def cnn_fcnn_model(features, model_params, example_description, training):
    # CNN part
    # Reshape inputs into proper dimensions
    for (name, f), d in zip(features.items(), example_description):
        if name == 'image':
            telescope_data = tf.reshape(f, [-1, *d['shape']], name='Image_reshape')

    # Load neural conv neural network model
    sys.path.append(model_params['model_directory'])
    cnn_network_module = importlib.import_module(model_params['cnn_fcnn']['cnn_network']['module'])
    cnn_network = getattr(cnn_network_module,
                          model_params['cnn_fcnn']['cnn_network']['function'])

    with tf.variable_scope("CNN"):
        output = cnn_network(telescope_data, params=model_params, training=training)
        cnn_output = tf.reduce_mean(output, axis=[1, 2], name='global_avgpool')
        #cnn_output = tf.layers.flatten(cnn_output, name='cnn_flatten')

    #FC after flat CNN
    try:
        sys.path.append(model_params['model_directory'])
        fcnn_network_module = importlib.import_module(model_params['cnn_fcnn']['nnAfterCNN_network']['module'])
        fcnn_network = getattr(fcnn_network_module,
                           model_params['cnn_fcnn']['nnAfterCNN_network']['function'])

        with tf.variable_scope("FC_after_CNN"):
            cnn_output = fcnn_network(cnn_output, params=model_params)
        print('you setted an FC NN after CNN')
    except:
        print('No FC after CNN will be used')
        pass
    if model_params['cnn_fcnn']['cnn_pretrained_weights']:    tf.contrib.framework.init_from_checkpoint(
        model_params['cnn_fcnn']['cnn_pretrained_weights'], {'CNN/': 'CNN/'})

    # FC part
    # Create a list with all tensor parameters and reshape each tensor as a column
    parameters_data_list = []
    for (name, f), d in zip(features.items(), example_description):
        if name.startswith('parameter_'):
            if not f.dtype.is_floating:
                f = tf.cast(f, dtype=tf.float32)
            parameters_data_tmp = tf.reshape(f, [-1, 1], name='Parameter_reshape')
            parameters_data_list.append(parameters_data_tmp)

    # Concatenate parameters on x axis
    parameters_data = parameters_data_list[0]
    for param_tensor in parameters_data_list[1:]:
        parameters_data = tf.concat([parameters_data, param_tensor], 1, name='Parameters_concat')

    # Load neural fully connected network model
    sys.path.append(model_params['model_directory'])
    fcnn_network_module = importlib.import_module(model_params['cnn_fcnn']['fcnn_network']['module'])
    fcnn_network = getattr(fcnn_network_module,
                           model_params['cnn_fcnn']['fcnn_network']['function'])

    with tf.variable_scope("FCNN"):
        fcnn_output = fcnn_network(parameters_data, params=model_params)
        # fcnn_output = tf.layers.flatten(fcnn_output, name='fcnn_flatten')

    if model_params['cnn_fcnn']['fcnn_pretrained_weights']:    tf.contrib.framework.init_from_checkpoint(
        model_params['cnn_fcnn']['fcnn_pretrained_weights'], {'FCNN/': 'FCNN/'})

    # concat the two flatten output
    with tf.variable_scope("Concatenate_Networks"):
        output = tf.keras.layers.concatenate([cnn_output, fcnn_output], axis=1)
    return output