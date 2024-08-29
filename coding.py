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
    return np.mean((Y - AL)**2)  # คำนวณ MSE ระหว่างค่าจริง (Y) กับค่าที่ทำนายได้ (AL)

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
    output_dim = Y.shape[1]  # จำนวนฟีเจอร์ในข้อมูลเอาท์พุต
    parameters = Initialize_Parameters(input_dim, hidden_layers, output_dim)  # สุ่มค่าพารามิเตอร์
    velocity = Initialize_Velocity(parameters)  # สุ่มค่าเทอมความเร็ว
    loss_per_epoch = []  # เก็บค่า loss ต่อ epoch
    percentage_loss_per_epoch = []  # เก็บค่าเปอร์เซ็นต์ loss ต่อ epoch
    
    for epoch in range(epochs):
        AL, caches = Forward_Propagation(X, parameters, hidden_layers)  # ทำ Forward Propagation
        loss = MSE_Loss(Y, AL)  # คำนวณค่า loss
        percentage_loss = Percentage_Loss(Y, AL)  # คำนวณค่าเปอร์เซ็นต์ loss
        loss_per_epoch.append(loss)  # เก็บค่า loss ต่อ epoch
        percentage_loss_per_epoch.append(percentage_loss)  # เก็บค่าเปอร์เซ็นต์ loss ต่อ epoch
        grads = Backward_Propagation(X, Y, parameters, caches, hidden_layers)  # ทำ Backward Propagation
        parameters, velocity = Update_Parameters(parameters, grads, velocity, learning_rate, momentum_rate)  # อัปเดตพารามิเตอร์
        
        if epoch % 100 == 0 or epoch == epochs - 1:  # แสดงผลลัพธ์ทุก ๆ 100 epochs หรือเมื่อถึง epoch สุดท้าย
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss} - Percentage Loss: {percentage_loss}%")
    
    # ทำนายค่าและคำนวณ loss สำหรับชุดทดสอบ
    Y_pred_test, _ = Forward_Propagation(X_test, parameters, hidden_layers)
    test_loss = MSE_Loss(Y_test, Y_pred_test)
    test_percentage_loss = Percentage_Loss(Y_test, Y_pred_test)
    print(f"Test Loss: {test_loss} - Test Percentage Loss: {test_percentage_loss}%")
    
    return loss_per_epoch, percentage_loss_per_epoch, parameters

# ฟังก์ชันสำหรับการทำ K-Fold Cross Validation
import numpy as np
import matplotlib.pyplot as plt

# Other functions (Read_Flood_data, Normalize, Sigmoid, etc.) remain unchanged

# Function for K-Fold Cross Validation
def KFold_CrossValidation(X, Y, hidden_layers, epochs, learning_rate, momentum_rate, K=10):
    fold_size = X.shape[0] // K  # ขนาดของแต่ละ fold
    losses = []  # เก็บค่า loss ของแต่ละ fold
    percentage_losses = []  # เก็บค่าเปอร์เซ็นต์ loss ของแต่ละ fold
    fold_scores = []  # เก็บค่า loss ต่อ epoch ของแต่ละ fold
    
    for k in range(K):
        print(f"Fold {k+1}/{K}")
        start, end = k * fold_size, (k + 1) * fold_size  # กำหนดขอบเขตของ fold ปัจจุบัน
        X_train = np.concatenate((X[:start], X[end:]), axis=0)  # ข้อมูลฝึกสอน
        Y_train = np.concatenate((Y[:start], Y[end:]), axis=0)  # ค่าฝึกสอน
        X_valid = X[start:end]  # ข้อมูลตรวจสอบ
        Y_valid = Y[start:end]  # ค่าตรวจสอบ

        loss_per_epoch, percentage_loss_per_epoch, _ = Train_MLP(X_train, Y_train, hidden_layers, epochs, learning_rate, momentum_rate, X_valid, Y_valid)
        losses.append(loss_per_epoch[-1])  # เก็บค่า loss สุดท้ายของ fold นี้
        percentage_losses.append(percentage_loss_per_epoch[-1])  # เก็บค่าเปอร์เซ็นต์ loss สุดท้ายของ fold นี้
        fold_scores.append(loss_per_epoch)  # เก็บ loss ต่อ epoch ของ fold นี้
    
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
    plt.bar(range(1, len(losses) + 1), losses, color='skyblue', edgecolor='black')
    plt.xlabel('Fold')
    plt.ylabel('Test MSE Loss')
    plt.title('Test MSE Loss for Each Fold')
    plt.xticks(range(1, len(losses) + 1))
    plt.show()
    
    return np.mean(losses), np.mean(percentage_losses)  # คืนค่าเฉลี่ยของ loss และเปอร์เซ็นต์ loss

if __name__ == "__main__":
    input_data, output_data = Read_Flood_data()  # อ่านข้อมูลจากไฟล์
    input_data, mean, std = Normalize(input_data)  # ทำ Normalization ข้อมูล

    hidden_layers = [8, 5]  # กำหนดจำนวน hidden layers
    epochs = 1000  # จำนวน epoch
    learning_rate = 0.01  # Learning rate
    momentum_rate = 0.9  # Momentum rate

    # ทำ K-Fold Cross Validation
    mean_loss, mean_percentage_loss = KFold_CrossValidation(input_data, output_data, hidden_layers, epochs, learning_rate, momentum_rate)
    print(f"Mean Loss: {mean_loss} - Mean Percentage Loss: {mean_percentage_loss}%")

    # Train the final model and get the loss per epoch for plotting
    loss_per_epoch, percentage_loss_per_epoch, _ = Train_MLP(input_data, output_data, hidden_layers, epochs, learning_rate, momentum_rate, input_data, output_data)

    # พล็อตกราฟ Loss ต่อ Epoch
    plt.plot(loss_per_epoch, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss over Epochs")
    plt.legend()
    plt.show()
