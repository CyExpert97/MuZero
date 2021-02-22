#!/usr/bin/env python

# import agents

import numpy as np
import tensorflow as tf

from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

from tensorflow.keras import backend as K

from huber import huber_loss_function, mean_squared_loss


#
# [Proximal Policy Optimization Algorithms,2017] (../../papers/ppo.pdf)
# improved
#

def get_ppo_actor_loss_clipped_obj(advantage_input, old_prediction_input, clip_epsilon=0.2, beta=0.01):
    """
        Make loss function for Proximal Policy optimization with clipped objective
        advantage = rewards - values
        current policy pi(at|st)
        second policy pi_theta(at|st)
        ratio between new and old policy
        according to mentioned paper, the clipping for surrogate loss is better then KL penalty.
        we implement only clipping.
        we get the minimum between standard surrogate loss and an epsilon clipped surrogate loss
        then we add entropy to the policy in order to improve exploration
        adding the entropy of the policy pi to the objective function improved exploration by discouraging premature
        convergence to suboptimal deterministic policies.
        This technique was originally proposed by (Williams & Peng, 1991),
        entropy_reg = beta * Gradient theta' ( H(pi(st;theta')) )
        H is entropy
        entropy  = pi * K.log(pi + epsilon)
    """

    def __actor_loss(y_true, y_pred):
        epsilon = 1e-10
        # epsilon = K.epsilon()  # 1e-7

        # p = y_pred

        # probability = K.sum(y_true * p, axis=1)

        policy = y_true * y_pred
        policy_old = y_true * old_prediction_input

        # add epsilon to avoid dividing by zero
        ratio = policy / (policy_old + epsilon)
        # K.clip may rise an error of unclear type - convert to tensor
        ratio = tf.convert_to_tensor(ratio)

        # clip ratio
        ratio_clipped = K.clip(
            ratio,
            min_value=1 - clip_epsilon,
            max_value=1 + clip_epsilon)

        # minimum between the standard surrogate loss and an epsilon clipped surrogate loss
        surrogate = K.minimum(ratio * advantage_input, ratio_clipped * advantage_input)

        # adding the entropy of the policy pi to the objective function improved exploration by discouraging premature
        # convergence to suboptimal deterministic policies.
        # This technique was originally proposed by (Williams & Peng, 1991),
        # entropy_reg = beta * Gradient theta' ( H(pi(st;theta')) )
        # H is entropy
        # entropy  = p * K.log(p + epsilon)
        entropy_reg = beta * K.sum(policy * K.log(policy + epsilon))

        policy_loss = -K.mean(surrogate)

        # input for the back propagation
        actor_loss = policy_loss + entropy_reg

        # alternative
        # entropy = beta *  policy * K.log(policy + epsilon)
        # actor_loss = -K.mean(surrogate + entropy)
        return actor_loss

    return __actor_loss


#
# [Proximal Policy Optimization Algorithms,2017] (../../papers/ppo.pdf)
# adapted to continuous control
#
# the Gaussian policy is described in paper
# [The Beta Policy for Continuous Control Reinforcement Learning, 2017](../../papers/thesis_chou.pdf)
#

def get_ppo_actor_loss_clipped_obj_continuous(advantage_input, old_prediction_input, clip_epsilon=0.2, sigma=1.0):
    '''
        Make loss function for Proximal Policy optimization with clipped objective
        advantage = rewards - values

        current policy pi(at|st)
        second policy pi_theta(at|st)
        pi_theta(a|s) = exp ( -square(a - mu) / (2 * square(sigma) ) )  /
                                                        ( sqrt(2 * np.pi ) * sigma )
        where the mean mu = mu_theta(s) (y_pred)
        and the standard deviation sigma = sigma_theta(s) is a noise
        a is y_true
        ratio between new and old policy
        according to mentioned paper, the clipping for surrogate loss is better then KL penalty.
        we implement only clipping.
        we get the minimum between standard surrogate loss and an epsilon clipped surrogate loss
    '''

    def __actor_loss(y_true, y_pred):
        epsilon = 1e-10
        # epsilon = K.epsilon()  # 1e-7

        # p = y_pred

        # probability = K.sum(y_true * p, axis=1)

        # print("get_ppo_actor_loss_clipped_obj_continuous")
        # print(type(y_true))
        # print(y_true.dtype)
        # print(y_true.get_shape())
        # print(type(y_pred))
        # print(y_pred.dtype)

        squared_noise = K.square(sigma)
        denominator = K.sqrt(2 * np.pi * squared_noise)

        # print(squared_noise.dtype)
        # print(denominator.dtype)

        if tf.is_tensor(y_true):
            policy = K.exp(- K.square(y_true - y_pred) / (2 * squared_noise)) / denominator
            policy_old = K.exp(- K.square(y_true - old_prediction_input) / (2 * squared_noise)) / denominator
        else:
            a_mu = tf.convert_to_tensor(y_true - y_pred, dtype=tf.float32)
            a_mu_old = tf.convert_to_tensor(y_true - old_prediction_input, tf.float32)
            policy = K.exp(- K.square(a_mu) / (2 * squared_noise)) / denominator
            policy_old = K.exp(- K.square(a_mu_old) / (2 * squared_noise)) / denominator

        # add epsilon to avoid dividing by zero
        ratio = policy / (policy_old + epsilon)

        # K.clip may rise an error of unclear type - convert to tensor
        ratio = tf.convert_to_tensor(ratio)

        # clip ratio
        ratio_clipped = K.clip(
            ratio,
            min_value=1 - clip_epsilon,
            max_value=1 + clip_epsilon)

        # minimum between the standard surrogate loss and an epsilon clipped surrogate loss
        surrogate = K.minimum(ratio * advantage_input, ratio_clipped * advantage_input)

        policy_loss = -K.mean(surrogate)

        # do not add entropy

        # input for the back propagation
        actor_loss = policy_loss

        return actor_loss

    return __actor_loss


def get_ppo_critic_loss(clip_value=np.inf):
    ''' mean of squared errors '''
    # def __critic_loss(y_true, y_pred):
    #    return K.mean(K.square(y_true - y_pred), axis=-1)
    # return __critic_loss
    return huber_loss_function(clip_value)


def test():
    shape = (6, 7)
    y_a = np.random.random(shape)
    y_b = np.random.random(shape)

    # advantage_input = Input(shape=(1,))

    # a_i = np.random.random( (1,6))
    a_i = np.random.random((1,))
    # a_i = 0.
    op_i = np.random.random(shape)

    loss = get_ppo_actor_loss_clipped_obj(a_i, op_i, clip_epsilon=0.2, beta=0.01)

    print("")
    print("================")
    print("get_ppo_actor_loss_clipped_obj")
    print("================")
    out1 = K.eval(loss(K.variable(y_a), K.variable(y_b)))
    print(out1)
    print("================")
    out2 = K.eval(loss(y_a, y_b))
    print(out2)
    print("================")

    loss = get_ppo_actor_loss_clipped_obj_continuous(a_i, op_i, clip_epsilon=0.2, sigma=1.0)

    print("")
    print("================")
    print("get_ppo_actor_loss_clipped_obj_continuous")
    print("================")
    out1 = K.eval(loss(tf.convert_to_tensor(y_a, dtype=tf.float32), tf.convert_to_tensor(y_b, dtype=tf.float32)))
    print(out1)
    print("================")
    out2 = K.eval(loss(y_a, y_b))
    print(out2)
    print("================")

    loss = get_ppo_critic_loss(np.inf)

    print("")
    print("================")
    print("get_ppo_critic_loss (inf)")
    print("================")
    out1 = K.eval(loss(K.variable(y_a), K.variable(y_b)))
    print(out1)
    print("================")
    out2 = K.eval(loss(y_a, y_b))
    print(out2)
    print("================")

    loss = get_ppo_critic_loss(1.0)
    print("")
    print("================")
    print("get_ppo_critic_loss (1.0)")
    print("================")
    out1 = K.eval(loss(K.variable(y_a), K.variable(y_b)))
    print(out1)
    print("================")
    out2 = K.eval(loss(y_a, y_b))
    print(out2)
    print("================")


if __name__ == "__main__":
    test()

    print("passed")
    input("Enter to exit")
