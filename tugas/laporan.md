NAMA    : ASRIZAL
KELAS   : RB
MATKUL  : TBD
NIM     : 121450010

Untuk memulai, unduh dataset CIFAR-10 dari Canadian Institute for Advanced Research. Dataset ini terdiri dari 60.000 gambar berwarna berukuran 32x32 piksel yang dikelompokkan dalam berbagai kategori objek. Setelah pengunduhan selesai, ekstrak folder yang berisi dataset CIFAR-10. Setelah memahami struktur dataset dan proses serialisasi menggunakan cPickle, langkah selanjutnya adalah menggunakan kode yang diberikan untuk membuka masing-masing dari lima file batch dan memuat semua gambar ke dalam array NumPy.

```python
import numpy as np
import pickle
from pathlib import Path

# Path to the unzipped CIFAR data
data_dir = Path("data/cifar-10-batches-py/")

# Unpickle function provided by the CIFAR hosts
def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict

images, labels = [], []
for batch in data_dir.glob("data_batch_*"):
    batch_data = unpickle(batch)
    for i, flat_im in enumerate(batch_data[b"data"]):
        im_channels = []
        # Each image is flattened, with channels in order of R, G, B
        for j in range(3):
            im_channels.append(
                flat_im[j * 1024 : (j + 1) * 1024].reshape((32, 32))
            )
        # Reconstruct the original image
        images.append(np.dstack((im_channels)))
        # Save the label
        labels.append(batch_data[b"labels"][i])

print("Loaded CIFAR-10 training set:")
print(f" - np.shape(images)     {np.shape(images)}")
print(f" - np.shape(labels)     {np.shape(labels)}")
```

Semua gambar sekarang tersimpan di memori dalam variabel gambar, lengkap dengan metadata yang sesuai dalam label.

selanjutnya hal yang perlu dilakukan adalah menginstal paket-paket Python yang diperlukan untuk melakukan tiga metode tersebut.

```shell
pip install Pillow
```
```shell
pip install lmdb
```
```shell
pip install h5py
```
### Menyimpan sebuah gambar

Sekarang, mari kita fokus pada perbandingan waktu yang diperlukan untuk membaca dan menulis file, serta jumlah memori disk yang digunakan untuk setiap tugas dasar yang menjadi perhatian kita.

Untuk keperluan eksperimen, kita akan membandingkan performa dengan berbagai jumlah file, mulai dari satu gambar hingga 100.000 gambar, dengan peningkatan bertahap sebesar faktor 10.

Sebagai persiapan untuk eksperimen ini, kita perlu membuat folder untuk masing-masing metode. Folder-foler ini akan berisi semua file database atau gambar, dan kita akan menyimpan jalur ke direktori-direktori tersebut dalam variabel.

```python
from pathlib import Path

disk_dir = Path("data/disk/")
lmdb_dir = Path("data/lmdb/")
hdf5_dir = Path("data/hdf5/")

disk_dir.mkdir(parents=True, exist_ok=True)
lmdb_dir.mkdir(parents=True, exist_ok=True)
hdf5_dir.mkdir(parents=True, exist_ok=True)
```
### Menyimpan ke dalam Disk

Dalam eksperimen ini, kita memiliki satu gambar yang saat ini disimpan di memori sebagai array NumPy. Tujuannya adalah menyimpan gambar tersebut ke disk sebagai file .png dengan nama unik yang ditentukan oleh ID gambar, yaitu image_id. Proses ini dapat dilakukan menggunakan paket Pillow yang telah diinstal sebelumnya.

```python
from PIL import Image
import csv

def store_single_disk(image, image_id, label):
    """ Stores a single image as a .png file on disk.
        Parameters:
        ---------------
        image       image array, (32, 32, 3) to be stored
        image_id    integer unique ID for image
        label       image label
    """
    Image.fromarray(image).save(disk_dir / f"{image_id}.png")

    with open(disk_dir / f"{image_id}.csv", "wt") as csvfile:
        writer = csv.writer(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        writer.writerow([label])
```

### Menyimpan ke dalam LMDB

LMDB adalah sistem penyimpanan kunci-nilai yang menyimpan setiap entri sebagai array byte. Dalam konteks ini, kunci berfungsi sebagai pengenal unik untuk setiap gambar, sedangkan nilai menyimpan gambar tersebut. Baik kunci maupun nilai diharapkan berbentuk string, sehingga nilai biasanya diserialisasi menjadi string saat penyimpanan dan dideserialisasi saat pembacaan kembali.

Untuk langkah ini, kita bisa membuat kelas dasar Python untuk merepresentasikan gambar beserta metadata-nya.

```python
class CIFAR_Image:
    def __init__(self, image, label):
        # Dimensions of image for reconstruction - not really necessary 
        # for this dataset, but some datasets may include images of 
        # varying sizes
        self.channels = image.shape[2]
        self.size = image.shape[:2]

        self.image = image.tobytes()
        self.label = label

    def get_image(self):
        """ Returns the image as a numpy array. """
        image = np.frombuffer(self.image, dtype=np.uint8)
        return image.reshape(*self.size, self.channels)
```
sekarang mari kita menyimpan satu gambar ke dalam LMDB:

```python
import lmdb
import pickle

def store_single_lmdb(image, image_id, label):
    """ Stores a single image to a LMDB.
        Parameters:
        ---------------
        image       image array, (32, 32, 3) to be stored
        image_id    integer unique ID for image
        label       image label
    """
    map_size = image.nbytes * 10

    # Create a new LMDB environment
    env = lmdb.open(str(lmdb_dir / f"single_lmdb"), map_size=map_size)

    # Start a new write transaction
    with env.begin(write=True) as txn:
        # All key-value pairs need to be strings
        value = CIFAR_Image(image, label)
        key = f"{image_id:08}"
        txn.put(key.encode("ascii"), pickle.dumps(value))
    env.close()
```
### Menyimpan ke dalam HDF5

Dalam kasus yang cukup sederhana ini, kita memiliki opsi untuk membuat dua dataset terpisah: satu untuk gambar dan satu lagi untuk meta datanya.

```python
import h5py

def store_single_hdf5(image, image_id, label):
    """ Stores a single image to an HDF5 file.
        Parameters:
        ---------------
        image       image array, (32, 32, 3) to be stored
        image_id    integer unique ID for image
        label       image label
    """
    # Create a new HDF5 file
    file = h5py.File(hdf5_dir / f"{image_id}.h5", "w")

    # Create a dataset in the file
    dataset = file.create_dataset(
        "image", np.shape(image), h5py.h5t.STD_U8BE, data=image
    )
    meta_set = file.create_dataset(
        "meta", np.shape(label), h5py.h5t.STD_U8BE, data=label
    )
    file.close()
```
h5py.h5t.STD_U8BE mengatur jenis data yang akan disimpan dalam dataset, yang dalam kasus ini adalah bilangan bulat tak bertanda dengan panjang 8 bit.

### Experiment untuk menyimpan satu gambar

pada proses ini kita menggabungkan ketiga fungsi untuk menyimpan sebuah gambar tunggal, yang kemudian dapat dipanggil kembali.

```python
_store_single_funcs = dict(
    disk=store_single_disk, lmdb=store_single_lmdb, hdf5=store_single_hdf5
)
```
selanjutkan kita akan mencoba menyimpan gambar pertama dari CIFAR beserta labelnya, dan menyimpannya dalam tiga cara yang berbeda:

```python
from timeit import timeit

store_single_timings = dict()

for method in ("disk", "lmdb", "hdf5"):
    t = timeit(
        "_store_single_funcs[method](image, 0, label)",
        setup="image=images[0]; label=labels[0]",
        number=1,
        globals=globals(),
    )
    store_single_timings[method] = t
    print(f"Method: {method}, Time usage: {t}")
```
output :

| Method | Time usage              |
|--------|-------------------------|
| disk   | 0.06597459997283295     |
| lmdb   | 0.011656099988613278    |
| hdf5   | 0.022266499989200383    |

Disk: Metode ini memiliki waktu penggunaan tertinggi, menunjukkan bahwa metode penyimpanan menggunakan disk memerlukan waktu lebih lama dibandingkan LMDB dan HDF5. Oleh karena itu, disk kurang efisien untuk penyimpanan gambar dalam konteks ini.

LMDB: Metode ini memiliki waktu penggunaan terendah, menjadikannya metode paling efisien untuk penyimpanan gambar. LMDB sangat cepat dalam melakukan operasi penyimpanan, menjadikannya pilihan terbaik di antara ketiga metode.

HDF5: Meskipun tidak secepat LMDB, HDF5 masih lebih efisien dibandingkan dengan metode disk. HDF5 adalah pilihan yang baik jika kecepatan yang sedikit lebih rendah dibandingkan LMDB masih dapat diterima, karena HDF5 menawarkan kelebihan lain seperti dukungan untuk data terstruktur kompleks.

### Experiment untuk menyimpan banyak gambar

pada proses ini kita perlu mengubah sedikit kode dan membuat fungsi baru yang menerima banyak gambar:

```python
store_many_disk(images, labels):
    """ Stores an array of images to disk
        Parameters:
        ---------------
        images       images array, (N, 32, 32, 3) to be stored
        labels       labels array, (N, 1) to be stored
    """
    num_images = len(images)

    # Save all the images one by one
    for i, image in enumerate(images):
        Image.fromarray(image).save(disk_dir / f"{i}.png")

    # Save all the labels to the csv file
    with open(disk_dir / f"{num_images}.csv", "w") as csvfile:
        writer = csv.writer(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        for label in labels:
            # This typically would be more than just one value per row
            writer.writerow([label])

def store_many_lmdb(images, labels):
    """ Stores an array of images to LMDB.
        Parameters:
        ---------------
        images       images array, (N, 32, 32, 3) to be stored
        labels       labels array, (N, 1) to be stored
    """
    num_images = len(images)

    map_size = num_images * images[0].nbytes * 10

    # Create a new LMDB DB for all the images
    env = lmdb.open(str(lmdb_dir / f"{num_images}_lmdb"), map_size=map_size)

    # Same as before — but let's write all the images in a single transaction
    with env.begin(write=True) as txn:
        for i in range(num_images):
            # All key-value pairs need to be Strings
            value = CIFAR_Image(images[i], labels[i])
            key = f"{i:08}"
            txn.put(key.encode("ascii"), pickle.dumps(value))
    env.close()

def store_many_hdf5(images, labels):
    """ Stores an array of images to HDF5.
        Parameters:
        ---------------
        images       images array, (N, 32, 32, 3) to be stored
        labels       labels array, (N, 1) to be stored
    """
    num_images = len(images)

    # Create a new HDF5 file
    file = h5py.File(hdf5_dir / f"{num_images}_many.h5", "w")

    # Create a dataset in the file
    dataset = file.create_dataset(
        "images", np.shape(images), h5py.h5t.STD_U8BE, data=images
    )
    meta_set = file.create_dataset(
        "meta", np.shape(labels), h5py.h5t.STD_U8BE, data=labels
    )
    file.close()
```
untuk langkah selanjutnya adalah menyiapkan dataset untuk experiment dengan meningkatkan ukurannya

```python
cutoffs = [10, 100, 1000, 10000, 100000]

# Let's double our images so that we have 100,000
images = np.concatenate((images, images), axis=0)
labels = np.concatenate((labels, labels), axis=0)

# Make sure you actually have 100,000 images and labels
print(np.shape(images))
print(np.shape(labels))
```
selanjutnya kita dapat membuat sebuah dictionary yang dapat mengelola semua fungsi dengan awalan store_many_ dan menjalankan experimennya

```python
_store_many_funcs = dict(
    disk=store_many_disk, lmdb=store_many_lmdb, hdf5=store_many_hdf5
)

from timeit import timeit

store_many_timings = {"disk": [], "lmdb": [], "hdf5": []}

for cutoff in cutoffs:
    for method in ("disk", "lmdb", "hdf5"):
        t = timeit(
            "_store_many_funcs[method](images_, labels_)",
            setup="images_=images[:cutoff]; labels_=labels[:cutoff]",
            number=1,
            globals=globals(),
        )
        store_many_timings[method].append(t)

        # Print out the method, cutoff, and elapsed time
        print(f"Method: {method}, Time usage: {t}")
```
lalu kita dapat membuat grafik untuk melihat hasil experiment ini :

```python
import matplotlib.pyplot as plt

def plot_with_legend(
    x_range, y_data, legend_labels, x_label, y_label, title, log=False
):
    """ Displays a single plot with multiple datasets and matching legends.
        Parameters:
        --------------
        x_range         list of lists containing x data
        y_data          list of lists containing y values
        legend_labels   list of string legend labels
        x_label         x axis label
        y_label         y axis label
    """
    plt.style.use("seaborn-whitegrid")
    plt.figure(figsize=(10, 7))

    if len(y_data) != len(legend_labels):
        raise TypeError(
            "Error: number of data sets does not match number of labels."
        )

    all_plots = []
    for data, label in zip(y_data, legend_labels):
        if log:
            temp, = plt.loglog(x_range, data, label=label)
        else:
            temp, = plt.plot(x_range, data, label=label)
        all_plots.append(temp)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(handles=all_plots)
    plt.show()

# Getting the store timings data to display
disk_x = store_many_timings["disk"]
lmdb_x = store_many_timings["lmdb"]
hdf5_x = store_many_timings["hdf5"]

plot_with_legend(
    cutoffs,
    [disk_x, lmdb_x, hdf5_x],
    ["PNG files", "LMDB", "HDF5"],
    "Number of images",
    "Seconds to store",
    "Storage time",
    log=False,
)

plot_with_legend(
    cutoffs,
    [disk_x, lmdb_x, hdf5_x],
    ["PNG files", "LMDB", "HDF5"],
    "Number of images",
    "Seconds to store",
    "Log storage time",
    log=True,
)
```

| Method | Time usage               |
|--------|--------------------------|
| disk   | 0.0247044000425376       |
| lmdb   | 0.009863700019195676     |
| hdf5   | 0.026961700001265854     |
| disk   | 0.10658550000516698      |
| lmdb   | 0.007389699982013553     |
| hdf5   | 0.002107099979184568     |
| disk   | 1.5134746000403538       |
| lmdb   | 0.06281229999149218      |
| hdf5   | 0.003965499985497445     |
| disk   | 13.027919800020754       |
| lmdb   | 0.41153069998836145      |
| hdf5   | 0.033163600019179285     |
| disk   | 130.50907390000066       |
| lmdb   | 4.191937300027348        |
| hdf5   | 0.558269300032407        |

Disk: Tidak efisien untuk skala besar karena waktu penggunaan yang meningkat secara signifikan.
LMDB: Efisien dan dapat diskalakan dengan baik, namun masih lebih lambat dibandingkan HDF5 untuk jumlah gambar yang sangat besar.

HDF5: Sangat efisien di semua skala, menunjukkan waktu penggunaan yang sangat rendah dan konsisten, menjadikannya pilihan terbaik untuk operasi penyimpanan gambar dalam jumlah besar.

Dari analisis ini, HDF5 muncul sebagai metode penyimpanan paling efisien, terutama saat menangani jumlah gambar yang besar. LMDB juga menunjukkan efisiensi yang baik, terutama bila dibandingkan dengan metode disk. Disk kurang efisien untuk skala besar dan sebaiknya dihindari untuk jumlah gambar yang sangat besar.

![Output1](/tugas/output1.png)

![Output2](/tugas/output2.png)

Grafik pertama memperlihatkan durasi penyimpanan standar yang tidak disesuaikan, yang menggambarkan perbedaan yang signifikan antara penyimpanan ke file .png dan LMDB atau HDF5.

pada grafik kedua kita dapat lihat bahwa durasi penyimpanan untuk HDF5 memulai dengan lebih cepat dan melambat secara signifikan jika dibandingkan dengan LMDB.

### Membaca Sebuah Gambar Tunggal

selanjutnya kita akan membaca sebuah gambar tunggal menggunakan tiga metode yang berbeda.

### Membaca dari Disk

Pertama, kita akan membaca sebuah gambar tunggal beserta metadatanya dari file .png dan .csv.

```python
def read_single_disk(image_id):
    """ Stores a single image to disk.
        Parameters:
        ---------------
        image_id    integer unique ID for image

        Returns:
        ----------
        image       image array, (32, 32, 3) to be stored
        label       associated meta data, int label
    """
    image = np.array(Image.open(disk_dir / f"{image_id}.png"))

    with open(disk_dir / f"{image_id}.csv", "r") as csvfile:
        reader = csv.reader(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        label = int(next(reader)[0])

    return image, label
```
### Membaca dari LMDB

```python
def read_single_lmdb(image_id):
    """ Stores a single image to LMDB.
        Parameters:
        ---------------
        image_id    integer unique ID for image

        Returns:
        ----------
        image       image array, (32, 32, 3) to be stored
        label       associated meta data, int label
    """
    # Open the LMDB environment
    env = lmdb.open(str(lmdb_dir / f"single_lmdb"), readonly=True)

    # Start a new read transaction
    with env.begin() as txn:
        # Encode the key the same way as we stored it
        data = txn.get(f"{image_id:08}".encode("ascii"))
        # Remember it's a CIFAR_Image object that is loaded
        cifar_image = pickle.loads(data)
        # Retrieve the relevant bits
        image = cifar_image.get_image()
        label = cifar_image.label
    env.close()

    return image, label
```
### Membaca dari HDF5

berikut adalah kode untuk membuka dan membaca file HDF5 serta mengurai gambar dan metadata yang sama:

```python
def read_single_hdf5(image_id):
    """ Stores a single image to HDF5.
        Parameters:
        ---------------
        image_id    integer unique ID for image

        Returns:
        ----------
        image       image array, (32, 32, 3) to be stored
        label       associated meta data, int label
    """
    # Open the HDF5 file
    file = h5py.File(hdf5_dir / f"{image_id}.h5", "r+")

    image = np.array(file["/image"]).astype("uint8")
    label = int(np.array(file["/meta"]).astype("uint8"))

    return image, label
```
selanjutnya kita dapat membuat sebuah dictionary yang berisi semua fungsi proses pembacaan.

```python
_read_single_funcs = dict(
    disk=read_single_disk, lmdb=read_single_lmdb, hdf5=read_single_hdf5
)
```
### Experiment untuk membaca sebuah gambar tunggal

berikut proses experiment yang dilakukan

```python
from timeit import timeit

read_single_timings = dict()

for method in ("disk", "lmdb", "hdf5"):
    t = timeit(
        "_read_single_funcs[method](0)",
        setup="image=images[0]; label=labels[0]",
        number=1,
        globals=globals(),
    )
    read_single_timings[method] = t
    print(f"Method: {method}, Time usage: {t}")
```
Berikut adalah hasil dari eksperimen untuk membaca sebuah gambar tunggal:

| Method | Time usage               |
|--------|--------------------------|
| disk   | 0.007421799993608147     |
| lmdb   | 0.007359899987932295     |
| hdf5   | 0.01380549999885261      |

Disk dan LMDB: Kedua metode ini menunjukkan kinerja yang sangat baik dengan waktu penggunaan yang rendah. LMDB sedikit lebih cepat daripada disk, menjadikannya pilihan terbaik di antara ketiga metode ini untuk kecepatan penyimpanan.

HDF5: Meskipun mungkin memiliki fitur atau kelebihan lain, dari segi waktu penggunaan, HDF5 lebih lambat dibandingkan dengan disk dan LMDB. Oleh karena itu, untuk aplikasi yang membutuhkan kecepatan tinggi, HDF5 mungkin bukan pilihan yang optimal.

Secara keseluruhan, jika kecepatan penyimpanan adalah faktor utama, LMDB adalah pilihan terbaik, diikuti oleh disk, dengan HDF5 sebagai pilihan terakhir dalam konteks kecepatan penyimpanan.

### Membaca banyak gambar

pada proses ini kita dapat membuat fungsi-fungsi dengan read_many_, yang akan digunakan untuk eksperimen berikutnya.

```python
def read_many_disk(num_images):
    """ Reads image from disk.
        Parameters:
        ---------------
        num_images   number of images to read

        Returns:
        ----------
        images      images array, (N, 32, 32, 3) to be stored
        labels      associated meta data, int label (N, 1)
    """
    images, labels = [], []

    # Loop over all IDs and read each image in one by one
    for image_id in range(num_images):
        images.append(np.array(Image.open(disk_dir / f"{image_id}.png")))

    with open(disk_dir / f"{num_images}.csv", "r") as csvfile:
        reader = csv.reader(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        for row in reader:
            labels.append(int(row[0]))
    return images, labels

def read_many_lmdb(num_images):
    """ Reads image from LMDB.
        Parameters:
        ---------------
        num_images   number of images to read

        Returns:
        ----------
        images      images array, (N, 32, 32, 3) to be stored
        labels      associated meta data, int label (N, 1)
    """
    images, labels = [], []
    env = lmdb.open(str(lmdb_dir / f"{num_images}_lmdb"), readonly=True)

    # Start a new read transaction
    with env.begin() as txn:
        # Read all images in one single transaction, with one lock
        # We could split this up into multiple transactions if needed
        for image_id in range(num_images):
            data = txn.get(f"{image_id:08}".encode("ascii"))
            # Remember that it's a CIFAR_Image object 
            # that is stored as the value
            cifar_image = pickle.loads(data)
            # Retrieve the relevant bits
            images.append(cifar_image.get_image())
            labels.append(cifar_image.label)
    env.close()
    return images, labels

def read_many_hdf5(num_images):
    """ Reads image from HDF5.
        Parameters:
        ---------------
        num_images   number of images to read

        Returns:
        ----------
        images      images array, (N, 32, 32, 3) to be stored
        labels      associated meta data, int label (N, 1)
    """
    images, labels = [], []

    # Open the HDF5 file
    file = h5py.File(hdf5_dir / f"{num_images}_many.h5", "r+")

    images = np.array(file["/images"]).astype("uint8")
    labels = np.array(file["/meta"]).astype("uint8")

    return images, labels

_read_many_funcs = dict(
    disk=read_many_disk, lmdb=read_many_lmdb, hdf5=read_many_hdf5
)
```
### Experiment untuk membaca banyak gambar

```python
from timeit import timeit

read_many_timings = {"disk": [], "lmdb": [], "hdf5": []}

for cutoff in cutoffs:
    for method in ("disk", "lmdb", "hdf5"):
        t = timeit(
            "_read_many_funcs[method](num_images)",
            setup="num_images=cutoff",
            number=1,
            globals=globals(),
        )
        read_many_timings[method].append(t)

        # Print out the method, cutoff, and elapsed time
        print(f"Method: {method}, No. images: {cutoff}, Time usage: {t}")
```

berikut hasil experiment yang dilakukan

![Output3](/tugas/output3.png)

![Output4](/tugas/output4.png)

| Method | No. images | Time usage            |
|--------|------------|-----------------------|
| disk   | 10         | 5.00003807246685e-07  |
| lmdb   | 10         | 3.00002284348011e-07  |
| hdf5   | 10         | 3.00002284348011e-07  |
| disk   | 100        | 2.00001522898674e-07  |
| lmdb   | 100        | 3.00002284348011e-07  |
| hdf5   | 100        | 2.00001522898674e-07  |
| disk   | 1000       | 3.00002284348011e-07  |
| lmdb   | 1000       | 4.00003045797348e-07  |
| hdf5   | 1000       | 2.00001522898674e-07  |
| disk   | 10000      | 2.00001522898674e-07  |
| lmdb   | 10000      | 2.00001522898674e-07  |
| hdf5   | 10000      | 3.9994483813643456e-07|
| disk   | 100000     | 2.00001522898674e-07  |
| lmdb   | 100000     | 3.00002284348011e-07  |
| hdf5   | 100000     | 3.00002284348011e-07  |

HDF5 umumnya menunjukkan kinerja yang baik, terutama saat menangani jumlah gambar yang lebih kecil (10, 100, 1000 gambar).
Disk sering menunjukkan kinerja yang cepat, terutama saat menangani jumlah gambar yang lebih besar (100000 gambar).
LMDB umumnya memiliki waktu penggunaan yang lebih lambat dibandingkan HDF5 dan disk dalam beberapa kasus, tetapi menunjukkan waktu yang konsisten baik untuk jumlah gambar kecil maupun besar.

Secara keseluruhan, pemilihan metode penyimpanan terbaik tergantung pada jumlah gambar yang akan diproses, dengan HDF5 sering menjadi pilihan terbaik untuk jumlah gambar yang lebih kecil dan disk untuk jumlah gambar yang lebih besar.



### Jumlah Memori yang terpakai untuk masing-masing metode

![Output5](/tugas/output5.png)

PNG: Metode ini tidak disarankan untuk penyimpanan gambar dalam jumlah besar karena waktu penyimpanannya yang meningkat secara signifikan. Namun, waktu pembacaannya tetap efisien.
LMDB dan HDF5: Kedua metode ini sangat efisien untuk penyimpanan dan pembacaan gambar dalam jumlah besar. Mereka menunjukkan kinerja yang jauh lebih baik daripada PNG dalam hal waktu penyimpanan, sementara tetap mempertahankan waktu pembacaan yang rendah dan konstan.

Secara keseluruhan, LMDB dan HDF5 adalah pilihan yang jauh lebih baik dibandingkan dengan PNG untuk tugas penyimpanan dan pembacaan gambar, terutama ketika berhadapan dengan jumlah gambar yang besar.





