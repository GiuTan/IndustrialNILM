from dependencies import *


training = False
os.environ["CUDA_VISIBLE_DEVICES"]="1"
approach = 'BERT'
data_path = 'processed_data/'
device = 'pelletizer'
type_= '_'+ approach +'_' + device
model_path, n_rows = params_loading(device,approach)
window_size = 360 # 360 * 10s = 3600 / 60 /  60 = 1 hour
output_dim = 2
setup_init()

print('DATA LOADING AND WINDOWING')
x_train, y_train, x_val, y_val, x_test, y_test, train_max = data_load_window(data_path, device, n_rows,window_size)
print('MODEL SETTING')
MODEL = model_setting(approach,model_path,training,type_,x_train,y_train,x_val,y_val,window_size,output_dim)
print('MODEL PREDICTION')
testing(MODEL,x_test,y_test,device,approach,train_max)



