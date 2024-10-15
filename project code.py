import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# قراءة البيانات من ملف Excel
data = pd.read_excel('C:/Users/alnaseem/OneDrive/Desktop/Ai course/lesson 7 (project machine learning)/project excel (train).xlsx')

# تحقق من القيم المفقودة
missing_values = data.isnull().sum()
print("Missing values in each column:\n", missing_values[missing_values > 0])

# ملء القيم المفقودة للأعمدة العددية فقط
num_data = data.select_dtypes(include='number')  # اختيار الأعمدة العددية فقط
data_filled = data.copy()  # عمل نسخة من البيانات الأصلية
data_filled[num_data.columns] = num_data.fillna(num_data.mean())  # ملء القيم المفقودة

# طباعة عدد الصفوف قبل وبعد التنظيف
print(f"Original data shape: {data.shape}")
print(f"Filled data shape: {data_filled.shape}")

# اختيار الأعمدة العددية (بعد ملء القيم المفقودة)
num_data_filled = data_filled.select_dtypes(exclude='object')

# تقسيم البيانات إلى مجموعة تدريب (90%) ومجموعة اختبار (10%)
if num_data_filled.shape[0] > 10:  # تأكد من وجود بيانات كافية
    X_train, X_test = train_test_split(num_data_filled, test_size=0.2, random_state=42)  # تغيير إلى 0.2
else:
    raise ValueError("Not enough data to split into train and test sets.")

# تطبيع البيانات
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# تطبيق K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_train_scaled)

# إضافة النتائج إلى مجموعة اختبار البيانات
X_test['Cluster'] = kmeans.predict(X_test_scaled)

# عرض بعض النتائج من مجموعة الاختبار
print(X_test.head())

# رسم النتائج
plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=X_test['Cluster'], cmap='viridis')
plt.title('K-Means Clustering Results on Test Data')
plt.xlabel(X_test.columns[0])
plt.ylabel(X_test.columns[1])
plt.show()

# وظيفة للتنبؤ بمجموعات جديدة
def predict_new_data(new_data):
    if len(new_data) != X_train.shape[1]:
        raise ValueError(f"Input must have {X_train.shape[1]} features, but got {len(new_data)}.")
    new_data_scaled = scaler.transform([new_data])
    cluster = kmeans.predict(new_data_scaled)
    return cluster[0]

# مثال على إدخال بيانات جديدة
new_input = [5, 3, 1, 0, 0, ...]  # استبدل القيم هنا بالقيم التي تتناسب مع الأعمدة العددية
predicted_cluster = predict_new_data(new_input)
print(f"The predicted cluster for the input {new_input} is: {predicted_cluster}")
