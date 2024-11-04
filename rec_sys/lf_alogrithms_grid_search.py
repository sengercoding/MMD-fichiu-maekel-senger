import lf_algorithms
from config import ConfigLf as config
from config import ConfigLf as config
import data_util as data
import jax.numpy as jnp
def gridsearch():
    #config.num_epochs=10
    ratings_tf, matrix_u, matrix_v, num_users, num_items = lf_algorithms.load_data_and_init_factors(config)
    train_ds, valid_ds, test_ds =  data.split_train_valid_test_tf(ratings_tf, config)
    grid_mse = []
     
    for lr in  [0.1 , 0.05, 0.01, 0.005, 0.001]:
        reg_mse = []
        for reg in [0.0001,0.0005, 0.001, 0.005 ,0.01  ] :
            
            print("lr: ",lr, "reg", reg)
            config.fixed_learning_rate= lr
            config.reg_param = reg


    

            # Optimize the factors fast
            matrix_u, matrix_v = lf_algorithms.uv_factorization_vec_reg(matrix_u, matrix_v, train_ds, valid_ds, config,log=False)
            mse_all_batches = lf_algorithms.mse_loss_all_batches(matrix_u, matrix_v, test_ds, config.batch_size_predict_with_mse)
            mse = jnp.mean(  jnp.array(mse_all_batches ))
            print("MSE on test_ds",mse)
            reg_mse.append(mse.item())
        grid_mse.append(reg_mse)
    print(grid_mse)

def comparison(lr,reg):
    ratings_tf, matrix_u, matrix_v, num_users, num_items = lf_algorithms.load_data_and_init_factors(config)
    train_ds, valid_ds, test_ds =  data.split_train_valid_test_tf(ratings_tf, config)

    matrix_u, matrix_v = lf_algorithms.uv_factorization_vec_no_reg(matrix_u, matrix_v, train_ds, valid_ds, config)
    mse_all_batches = lf_algorithms.mse_loss_all_batches(matrix_u, matrix_v, test_ds, config.batch_size_predict_with_mse)
    mse_og = jnp.mean(  jnp.array(mse_all_batches ))
    print(" MSE of original method on test_ds",mse_og)

    print("Regularized with reg=",reg,"and fixed learning rate =",lr)
    config.fixed_learning_rate= lr
    config.reg_param = reg


    
    
    # Optimize the factors fast
    matrix_u, matrix_v = lf_algorithms.uv_factorization_vec_reg(matrix_u, matrix_v, train_ds, valid_ds, config )
    mse_all_batches = lf_algorithms.mse_loss_all_batches(matrix_u, matrix_v, test_ds, config.batch_size_predict_with_mse)
    mse_reg = jnp.mean(  jnp.array(mse_all_batches ))
    print("Regularized MSE on test_ds",mse_reg)

# Test the functions
if __name__ == '__main__':
    
    
    #gridsearch() #best
    comparison(0.05, 0.001)