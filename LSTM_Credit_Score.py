import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd


file_path = input("Enter the file name +.csv : ")
data = pd.read_csv(file_path)
user_ids = data['user_id'].astype(int)
income = data['income'].astype(float)
credit_history = data['credit_history']
debt_to_income_ratio = data['debt_to_income_ratio'].astype(float)
age = data['age'].astype(int)
target = data['target']


credit_history_mapping = {'poor': 0, 'good': 1, 'excellent': 2}
target_mapping = {'low': 0, 'medium': 1, 'high': 2}

credit_history_numeric = np.array([credit_history_mapping[ch] for ch in credit_history])
target_numeric = np.array([target_mapping[t] for t in target])


user_ids_tensor = torch.LongTensor(user_ids.values)
income_tensor = torch.FloatTensor(income.values)
credit_history_tensor = torch.LongTensor(credit_history_numeric)
debt_to_income_ratio_tensor = torch.FloatTensor(debt_to_income_ratio.values)
age_tensor = torch.LongTensor(age.values)
target_tensor = torch.LongTensor(target_numeric)


selected_columns = ['income', 'debt_to_income_ratio', 'age']
correlation_matrix = data[selected_columns].corr().values.astype(np.float32)

# Преобразуване на данните за LSTM
X_tensor = torch.stack([income_tensor, credit_history_tensor, debt_to_income_ratio_tensor, age_tensor], dim=1)
X_tensor = X_tensor.view(X_tensor.shape[0], 1, -1)

# Добавяне на корелационната матрица като допълнителен признак
correlation_matrix_tensor = torch.FloatTensor(correlation_matrix)
correlation_matrix_tensor = correlation_matrix_tensor.unsqueeze(0).expand(X_tensor.size(0), -1, -1)

# Преобразуване на размерностите на тензорите
X_tensor = X_tensor.view(X_tensor.size(0), -1, 1)
correlation_matrix_tensor = correlation_matrix_tensor.view(correlation_matrix_tensor.size(0), -1, 1)


X_tensor = torch.cat([X_tensor, correlation_matrix_tensor], dim=1)


# дефиниране на LSTM
class CustomLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, correlation_matrix_size):
        super(CustomLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size + correlation_matrix_size, 50)
        self.fc2 = nn.Linear(50, output_size)

    def forward(self, x, correlation_matrix):
        out, _ = self.lstm(x)
        out = torch.cat([out[:, -1, :], correlation_matrix.squeeze(2)], dim=1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


# Инициализиране на модела
input_size = X_tensor.shape[2]
hidden_size = 50
correlation_matrix_size = correlation_matrix_tensor.size(1)
output_size = len(target_mapping)


model = CustomLSTMModel(input_size, hidden_size, output_size, correlation_matrix_size)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение на модела
num_epochs = int(input("Enter the number of epochs: "))
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Provide both X_tensor and correlation_matrix_tensor to the model
    outputs = model(X_tensor, correlation_matrix_tensor)
    loss = criterion(outputs, target_tensor.view(-1))

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Оценка на модела
model.eval()
with torch.no_grad():
    test_outputs = model(X_tensor, correlation_matrix_tensor)
    _, predicted_labels = torch.max(test_outputs, 1)


predicted_target_text = []
for label in predicted_labels.numpy():
    predicted_target_text.append([k for k, v in target_mapping.items() if v == label][0])

# Извеждане на резултатите
for i in range(len(user_ids)):
    print(f"Потребител {user_ids[i]} - Предсказан кредитен риск: {predicted_target_text[i]}")

count=0
correctPrediction=0
for i in range(len(user_ids)):
    count+=1
    if predicted_target_text[i]==target[i]:
        correctPrediction+=1
print(f"Успеваемост на познаване: {correctPrediction}/{count}")
print(f"Процент на успеваемост: {correctPrediction*100//count}%")