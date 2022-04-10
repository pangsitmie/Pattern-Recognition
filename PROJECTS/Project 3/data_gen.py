import csv
import random
import time
from kmeans_sample import KMeans

x_value = 0
center_1 = 0
center_2 = 0

fieldnames = ["x_value", "center_1", "center_2"]


with open('data.csv', 'w') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

while True:

    with open('data.csv', 'a') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        info = {
            "x_value": x_value,
            "center_1": center_1,
            "center_2": center_2
        }

        csv_writer.writerow(info)
        print(x_value, center_1, center_2)

        x_value += 1
        center_1 = k = KMeans(K=2, max_iters = 150, plot_steps=True)
        total_2 = total_2 + random.randint(-5, 6)

    time.sleep(1)