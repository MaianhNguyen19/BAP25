import csv
import os

path = r"C:\Users\harim\Desktop\grip_code\BAP25\Measurement code\Measurements"
foldername = "Measurements"
filename = "only_testing.csv"

file_path = os.path.join(path, filename)

print("Saving to:", file_path)
with open(file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['test1', 'test2', 'test3'])
