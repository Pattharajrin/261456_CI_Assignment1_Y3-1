import numpy as np
import matplotlib.pyplot as plt

def Read_Flood_data():
    with open("Flood_dataset.txt", "r") as f:
        data = [line.split() for line in f]

    data = np.array(data[2:])  # ข้ามสองบรรทัดแรก
    input_train = np.array([i[:8] for i in data], dtype=np.float32)  # ใช้ list comprehension
    output_train = np.array([i[-1] for i in data], dtype=np.int64)  # ใช้ list comprehension
    output_train = output_train.reshape(-1, 1)  # แปลงให้เป็นสองมิติ

    return input_train, output_train  # คืนค่า input_train และ output_train

def Read_Cross_data(filename='cross.txt'):
    data = []
    with open(filename) as f:
        lines = f.readlines()
        for line in range(1, len(lines), 3):
            cross1 = np.array([float(element) for element in lines[line].strip().split()])
            cross2 = np.array([float(element) for element in lines[line + 1].strip().split()])
            data.append(np.hstack((cross1, cross2)))  # ใช้ hstack แทน np.append

    data = np.array(data)
    input_data = data[:, :-2]  # แยก input
    output_data = data[:, -2:]  # แยก output
    return input_data, output_data  # คืนค่า input_data และ output_data

def Normalize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_normalized = (X - mean) / std
    return X_normalized, mean, std

def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

def Sigmoid_Derivative(x):
    return x * (1 - x)

# Initialize parameters
def Initialize_Parameters(input_layers, hidden_layers, output_layers):
    parameters = {}
    layer_dims = [input_layers] + hidden_layers + [output_layers]
    
    # เริ่มต้นพารามิเตอร์ weights และ biases สำหรับแต่ละชั้นในเครือข่ายประสาทเทียม
    for l in range(1, len(layer_dims)):
        parameters[f"weights{l}"] = np.random.randn(layer_dims[l-1], layer_dims[l]) * 0.01  # สุ่มค่า weights โดยคูณด้วย 0.01 เพื่อลดความเป็นไปได้ที่ค่าจะเริ่มต้นสูงเกินไป
        parameters[f"biases{l}"] = np.zeros((1, layer_dims[l]))  # เริ่มต้นค่า biases ด้วยค่า 0
    
    return parameters

# Initialize velocity terms
def Initialize_Velocity(parameters):
    velocity = {}
    L = len(parameters) // 2  # จำนวนชั้นของเครือข่ายประสาทเทียม (ไม่นับ input layer)
    
    # เริ่มต้นค่าเทอมความเร็ว (velocity) สำหรับแต่ละพารามิเตอร์ weights และ biases
    for l in range(1, L + 1):
        velocity[f"dweights{l}"] = np.zeros_like(parameters[f"weights{l}"])  # เริ่มต้นเทอมความเร็วของ weights ด้วยค่า 0
        velocity[f"dbiases{l}"] = np.zeros_like(parameters[f"biases{l}"])  # เริ่มต้นเทอมความเร็วของ biases ด้วยค่า 0
    
    return velocity

def Forward_Propagation(x, parameters, hidden_layers):
    caches = {}  # Create a variable to store values needed for backpropagation
    A = x  # Set the initial A to the input X
    L = len(hidden_layers) + 1  # Total number of layers in the model (including output layer)

    # Forward propagation through all hidden layers
    for l in range(1, L):
        Z = np.dot(A, parameters[f"weights{l}"]) + parameters[f"biases{l}"]  # Calculate Z = W·A + b
        A = Sigmoid(Z)  # Pass Z through the sigmoid function to get A
        caches[f"Z{l}"] = Z  # Store Z in caches
        caches[f"A{l}"] = A  # Store A in caches

    # Forward propagation for the last layer (output layer)
    ZL = np.dot(A, parameters[f"weights{L}"]) + parameters[f"biases{L}"]  # Calculate Z for the last layer
    AL = ZL  # For the last layer, A equals Z (no activation function used here)
    caches[f"Z{L}"] = ZL  # Store Z for the last layer in caches
    caches[f"A{L}"] = AL  # Store A for the last layer in caches

    return AL, caches  # Return final output (AL) and caches for use in backpropagation

# ฟังก์ชันคำนวณค่า Mean Squared Error (MSE) loss
def MSE_Loss(Y, AL):
    return (np.mean((Y - AL))**2)  # คำนวณ MSE ระหว่างค่าจริง (Y) กับค่าที่ทำนายได้ (AL)

# ฟังก์ชันคำนวณค่าเปอร์เซ็นต์ความสูญเสีย
def Percentage_Loss(Y, AL):
    return np.mean(np.abs((Y - AL) / Y)) * 100  # คำนวณเปอร์เซ็นต์ความสูญเสียระหว่างค่าจริงกับค่าที่ทำนาย


def Backward_Propagation(X, Y, parameters, caches, hidden_layers):
    grads = {}  # สร้างดิกชันนารีเพื่อเก็บค่าเกรดต่างๆ
    m = X.shape[0]  # จำนวนตัวอย่างในชุดข้อมูล
    L = len(hidden_layers) + 1  # จำนวนชั้นทั้งหมดในโมเดล (รวม output layer)
    
    # สำหรับเลเยอร์สุดท้าย
    AL = caches[f"A{L}"]
    dZL = AL - Y  # ใช้ AL แทนที่ caches[f"A{L}"]
    
    # คำนวณ gradients สำหรับเลเยอร์สุดท้าย
    grads[f"dweights{L}"] = np.dot(caches[f"A{L-1}"].T, dZL) / m
    grads[f"dbiases{L}"] = np.sum(dZL, axis=0, keepdims=True) / m

    # สำหรับเลเยอร์อื่นๆ
    for l in reversed(range(1, L)):
        dA_prev = np.dot(dZL, parameters[f"weights{l+1}"].T)
        dZ = dA_prev * Sigmoid_Derivative(caches[f"A{l}"])
        grads[f"dweights{l}"] = np.dot(caches[f"A{l-1}"].T, dZ) / m if l > 1 else np.dot(X.T, dZ) / m
        grads[f"dbiases{l}"] = np.sum(dZ, axis=0, keepdims=True) / m
        dZL = dZ
    
    return grads  # คืนค่าเกรดทั้งหมดที่คำนวณได้

# ฟังก์ชันอัปเดตพารามิเตอร์
def Update_Parameters(parameters, grads, velocity, learning_rate, momentum_rate):
    L = len(parameters) // 2  # จำนวนชั้นในโมเดล (ไม่นับ input layer)
    
    for l in range(1, L + 1):  # วนลูปผ่านทุกชั้น
        # ตรวจสอบขนาดของ grads กับ velocity เพื่อป้องกันข้อผิดพลาด
        assert grads[f"dweights{l}"].shape == velocity[f"dweights{l}"].shape, "Shape mismatch in dweights"
        assert grads[f"dbiases{l}"].shape == velocity[f"dbiases{l}"].shape, "Shape mismatch in dbiases"

        # คำนวณค่า velocity สำหรับ weights โดยใช้ Momentum
        velocity[f"dweights{l}"] = momentum_rate * velocity[f"dweights{l}"] + (1 - momentum_rate) * grads[f"dweights{l}"]
        # คำนวณค่า velocity สำหรับ biases โดยใช้ Momentum
        velocity[f"dbiases{l}"] = momentum_rate * velocity[f"dbiases{l}"] + (1 - momentum_rate) * grads[f"dbiases{l}"]
        
        # อัปเดตค่า weights ด้วยค่า velocity
        parameters[f"weights{l}"] -= learning_rate * velocity[f"dweights{l}"]
        # อัปเดตค่า biases ด้วยค่า velocity
        parameters[f"biases{l}"] -= learning_rate * velocity[f"dbiases{l}"]
    
    return parameters, velocity  # คืนค่า parameters และ velocity ที่อัปเดตแล้ว

def Train_MLP(X, Y, hidden_layers, epochs, learning_rate, momentum_rate, X_test, Y_test):
    input_dim = X.shape[1]  # จำนวนฟีเจอร์ในข้อมูลอินพุต
    output_dim = Y.shape[1]  # จำนวนฟีเจอร์ในข้อมูลเอาต์พุต
    parameters = Initialize_Parameters(input_dim, hidden_layers, output_dim)  # เริ่มต้นพารามิเตอร์ของโมเดล
    velocity = Initialize_Velocity(parameters)  # เริ่มต้นค่า velocity
    
    epoch_losses = []  # รายการสำหรับเก็บค่า loss ในแต่ละยุค
    
    for epoch in range(epochs):  # วนลูปผ่านจำนวนยุคที่กำหนด
        AL, caches = Forward_Propagation(X, parameters, hidden_layers)  # คำนวณ forward propagation
        loss = MSE_Loss(Y, AL)  # คำนวณค่า Mean Squared Error (MSE) loss
        percent_loss = Percentage_Loss(Y, AL)  # คำนวณเปอร์เซ็นต์ความสูญเสีย
        grads = Backward_Propagation(X, Y, parameters, caches, hidden_layers)  # คำนวณ gradients
        parameters, velocity = Update_Parameters(parameters, grads, velocity, learning_rate, momentum_rate)  # อัปเดตพารามิเตอร์
        
        # ประเมินโมเดลด้วยชุดข้อมูลทดสอบ
        AL_test, _ = Forward_Propagation(X_test, parameters, hidden_layers)  # คำนวณ forward propagation สำหรับชุดข้อมูลทดสอบ
        test_loss = MSE_Loss(Y_test, AL_test)  # คำนวณค่า loss สำหรับชุดข้อมูลทดสอบ
        epoch_losses.append(test_loss)  # เก็บค่า loss ของชุดข้อมูลทดสอบในแต่ละยุค
        
        # แสดงผลการฝึกในแต่ละยุค
        if epoch % 10 == 0:  # แสดงผลทุก ๆ 10 ยุค
            print(f"Epoch {epoch}, Loss: {loss:.4f}, Test Loss: {test_loss:.4f}, Percent Loss: {percent_loss:.2f}%")
    
    return parameters, epoch_losses  # คืนค่าพารามิเตอร์และค่า loss ในแต่ละยุค

def K_Fold_Cross_Validation(X, Y, K=10):
    fold_size = len(X) // K  # ขนาดของแต่ละ fold
    indices = np.random.permutation(len(X))  # สุ่มลำดับของข้อมูล

    fold_losses = []  # รายการสำหรับเก็บค่า loss ในแต่ละ fold
    fold_scores = []  # รายการสำหรับเก็บค่า scores ในแต่ละ fold
    final_parameters = None
    final_loss = None
    final_percent_loss = None
    last_X_train, last_Y_train, last_X_test, last_Y_test = None, None, None, None

    for i in range(K):
        start, end = i * fold_size, (i + 1) * fold_size
        test_indices = indices[start:end]  # ดัชนีสำหรับชุดทดสอบ
        train_indices = np.concatenate((indices[:start], indices[end:]))  # ดัชนีสำหรับชุดฝึก

        X_train, Y_train = X[train_indices], Y[train_indices]  # ข้อมูลฝึก
        X_test, Y_test = X[test_indices], Y[test_indices]  # ข้อมูลทดสอบ

        # Train the model
        hidden_layers = [10, 5]  # โครงสร้างของ hidden layers
        epochs = 100  # จำนวนยุค
        learning_rate = 0.0001  # อัตราการเรียนรู้
        momentum_rate = 0.9  # อัตรา momentum
        parameters, epoch_losses = Train_MLP(X_train, Y_train, hidden_layers, epochs, learning_rate, momentum_rate, X_test, Y_test)

        # Evaluate the model
        AL_test, _ = Forward_Propagation(X_test, parameters, hidden_layers)  # คำนวณค่า AL สำหรับชุดทดสอบ
        test_loss = MSE_Loss(Y_test, AL_test)  # คำนวณค่า MSE loss
        test_percent_loss = Percentage_Loss(Y_test, AL_test)  # คำนวณเปอร์เซ็นต์ความสูญเสีย

        fold_losses.append(test_loss)  # เก็บค่า loss ของ fold นี้
        fold_scores.append(epoch_losses)  # เก็บค่า scores ของ fold นี้

        if i == K - 1:
            final_parameters = parameters
            final_loss = test_loss
            final_percent_loss = test_percent_loss
            last_X_train, last_Y_train, last_X_test, last_Y_test = X_train, Y_train, X_test, Y_test

        print(f"Fold {i+1}, Test MSE Loss: {test_loss:.4f}, Test Percent Loss: {test_percent_loss:.2f}%")

    avg_loss = np.mean(fold_losses)  # ค่าเฉลี่ยของ loss ในทุก fold
    avg_percent_loss = np.mean([Percentage_Loss(Y_test, Forward_Propagation(X_test, final_parameters, hidden_layers)[0]) for test_indices in fold_losses])  # คำนวณค่าเฉลี่ยเปอร์เซ็นต์ความสูญเสีย
    print(f"Average Test MSE Loss: {avg_loss:.4f}, Average Test Percent Loss: {avg_percent_loss:.2f}%")

    return final_parameters, final_loss, final_percent_loss, last_X_train, last_Y_train, last_X_test, last_Y_test, fold_scores, fold_losses, avg_loss, avg_percent_loss

X, Y = Read_Flood_data()  # อ่านข้อมูลจากไฟล์
X_normalized, mean, std = Normalize(X)  # ปรับขนาดข้อมูล
# Ensure K-Fold returns the correct values
final_parameters, final_loss, final_percent_loss, last_X_train, last_Y_train, last_X_test, last_Y_test, fold_scores, fold_losses, avg_loss, avg_percent_loss = K_Fold_Cross_Validation(X_normalized, Y, K=10)

# Perform forward propagation using the last training data
AL_train, _ = Forward_Propagation(last_X_train, final_parameters, [10, 5])  # Calculate predictions for training data
AL_test, _ = Forward_Propagation(last_X_test, final_parameters, [10, 5])    # Calculate predictions for testing data

# ตรวจสอบค่าที่คำนวณได้
print("AL_train:", AL_train)
print("AL_test:", AL_test)

# วาดกราฟผลลัพธ์สำหรับข้อมูลการฝึก
plt.figure(figsize=(10, 5))
plt.plot(last_Y_train, label='Real Data (Train)', color='blue')
plt.plot(AL_train, label='Predicted Data (Train)', color='red')
plt.legend()
plt.title(f'Training Data - Final MSE Loss: {final_loss:.7f}, % Loss: {final_percent_loss:.2f}%, AVG Loss: {avg_loss:.2f}, AVG % Loss: {avg_percent_loss:.2f}%')
plt.show()

# Plot Test MSE Loss over epochs for each fold
plt.figure(figsize=(10, 5))
colors = plt.cm.viridis(np.linspace(0, 1, len(fold_scores)))
for i, epoch_losses in enumerate(fold_scores):
    print(f"Fold {i+1} - Epoch Losses Length: {len(epoch_losses)}")
    plt.plot(range(len(epoch_losses)), epoch_losses, color=colors[i], label=f'Fold {i+1}')
plt.xlabel('Epoch')
plt.ylabel('Test MSE Loss')
plt.title('Test MSE Loss vs Epochs for Each Fold')
plt.legend()
plt.show()

# Plot final Test MSE Loss for each fold
plt.figure(figsize=(10, 5))
plt.bar(range(1, len(fold_losses) + 1), fold_losses, color='skyblue', edgecolor='black')
plt.xlabel('Fold')
plt.ylabel('Test MSE Loss')
plt.title('Test MSE Loss for Each Fold')
plt.xticks(range(1, len(fold_losses) + 1))
plt.show()
