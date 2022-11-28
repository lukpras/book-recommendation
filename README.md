# Sistem Rekomendasi Buku menngunakan ***Content Based*** dan ***Collaborative Filtering*** - Luki Prasetyo

## **Project Overview**

Proyek ini merupakan sebuah proyek untuk membuat sistem rekomendasi berbasis kolaborasi dan berbasi konten. Membeli barang secara online telah menjadi tren saat ini, dibandingkan dengan pergi keluar dan membeli barang sendiri. Ketika membeli barang secara online maka, tedapat salah satu cara bagi penjual atau penyedia layanan belanja online untuk meningkatkan minat beli dari pembeli, salah satu caranya adalah dengan memberi rekomendasi barang, sehingga pembeli menjadi antusias, memudahkan pembeli untuk mencari barang lain dan memberikan cara yang lebih mudah dan cepat untuk membeli barang.
<br />
<br />
Karena di era saat ini yang kebanyakan kegiatan dapat dilakukan secara online, maka sistem rekomendasi dapat digunakan untuk meningkatkan kepuasan pelanggan dan juga meningkatkan aktivitas user, yang dapat berperan penting dalam membuat instansi atau perusahaan untuk meningkatkan kepuasan pelanggan dan juga meningkatkan pendapatan perusahaan, dengan menawarkan rekomendasi yang tepat bagi pelanggan/user. Sehingga sistem rekomendasi sangatlah penting, dimana pada kasus ini adalah rekomendasi untuk buku yang dibaca oleh pelanggan/user.
<br />
<br />
Permasalah tersebut dapat diatasi dengan melakukan pembuatan *machine learning* yang dapat melakukan rekomendasi berdasrkan beberapa fitur dari buku dengan melakukan pembuatan rekomendasi berdasrkan penulis buku dan rating buku menggunakan rekomendasi berbasi konten (*content based*) dan berbasis rating dari pelanggan/user menngunakan filter kolaborasi (*collaborative filtering*).
<br />
<br />
Variabel yang digunakan pada sistem rekomendasi berbasis konten (*content based*) adalah:
- Judul Buku
- Penulis Buku
- Rating Buku
Dan variabel yang digunakan pada sistem rekomendasi berbasis kolaborasi (*collaborative filtering*) adalah:
- Rating buku
- User/pelanggan

Adapun referensi yang digunakan dalam pembuatan proyek ini yaitu:

Jurnal Pertama
- Jurnal: ***Book Recommendation System through content based and collaborative filtering method***
- Authors: P. Mathew, B. Kuriakose dan V. Hegde
- *International Conference on Data Mining and Advanced Computing (SAPIENCE)*
- Di Terbitkan pada 12 Desember 2016

Jurnal Kedua
- Jurnal: ***Machine Learning based Efficient Recommendation System for Book Selection using User based Collaborative Filtering Algorithm***
- Authors: M. Kommineni, P. Alekhya, T. M. Vyshnavi, V. Aparna, K. Swetha and V. Mounika
- *2020 Fourth International Conference on Inventive Systems and Control (ICISC)*
- Di Terbitkan pada 19 Agustuws 2020


<br/><br/>

## Business Understanding

Pada era saat ini dimana banyak kegiatan dapat dilakukan secara online, dimulai dari membaca, memebli barang, berinteraksi dan kegiatan lain. Data dari kegiatan tersebut dapat menjadi salah satu resource dalam menentukan keputusan atau juga dapat disebut sebagai data-driven decision-making, maka dperlukan alat untuk mengolah data tersebut sehingga menghasilkan nilai, salah satunya dengan menggunakan algoritma-algoritma di machine learning.

### Problem Statements

Seperti yang telah disebutkan dalam latar belakang permasalahan, maka terdapat beberapa masalah yang harus dipecahkan, yaitu:
- Bagaimana cara membuat sistem rekomendasi agar sistem rekomendasi tersebut dapat digunakan dalam memberikan rekomendasi yang sesuai dengan pelanggan/user.
- Algoritma mana yang tepat untuk melakukan pembuatan sistem rekomendasi.
- Fitur-fitur mana yang tepat untuk membuat sistem rekomendasi.

### Goals

Setelah permasalahan diketahui, maka harus memiliki tujuan dalam menyelesaikan permasalah tersebut, dan permasalah tersebut dapat diselesaikan dengan:
- Membuat sistem rekomendasi menggunakan *machine learning* dengan bahasa pemrograman python.
- Memilih algoritma yang tepat dalam membuat rekomendasi berbasis konten dan berbasis rating dari pelanggan/user.
- Memilih fitur-fitur yang baik dan sesuai dengan algoritma yang digunakan, untuk menghasilkan sistem rekomendasi yang baik dan tepat.

### Solution statements

- Memilih algoritma sistem rekomendasi berbasis konten dari buku dan personalisasi dari pelanggan.
- Menggunakan algoritma *Content Based* dan *Collaborative Filtering* dalam penyelesaian permasalahan yang ada.
- Menggunakan algoritma *Content Based* dengan bebrapa proses preparasi berupa membuat *bag of words*, vektoriasasi kata, dan menggunakan ukuran kesamaan beruapa *cosine similarity*, dan untuk algoritma *Collaborative Filtering* menggunakan RecommenderNet, dan memilih fitur rating sebagai fitur utama pada model.

<br/><br/>

## Data Understanding
Data set yang digunakan merupakan data yang berada pada repository github yang dimiliki oleh **zygmuntz**, dengan nama data set yaitu **"goodbooks-10k"**.
Data set tersebut merupakan data yang di **scraping** dari [https://www.goodreads.com/](https://www.goodreads.com). data set ini memiliki 6 juta rating untuk 10 ribu buku paling populer.

Informasi data sebagai berikut:
- Rating buku bernilai antara 1 sampai dengan 5.
- Data set *books.csv* memiliki 10 ribu *book_id*, dan data set *ratings.csv* memiliki 53424 *user_id*, dimana setiap user setidaknya telah melakukan rating untuk dua buku.
- Pada data set ini pun juga memiliki variabel-variabel berupa, buku yang telah di tandai untuk dibaca, metadata dari buku (author, tahun, dll) dan juga tag dari buku.
- Pada data set *to_read.csv* memiliki *book_id* dari setiap buku yang telah ditandai oleh *user_id*.
- *book_tags.csv* memiliki nilai *tags* yang telah di masukan oleh user, yang disimpan dalam *tag_id*.
- *tags.csv* menerjemahkan *tag_id* menjadi nama dari *tags*.

Sumber data dapat diakses pada [goodbooks-10k repositroy oleh zygmuntz](https://github.com/zygmuntz/goodbooks-10k)

### Variabel-variabel pada goodbooks-10k dataset yang digunakan dalam pembuatan sistem rekomendasi adalah sebagai berikut:
- books.csv
    - id: id unik dari dataset (integer)
    - book_id: id dari buku pada situs **goodreads** (integer)
    - best_book_id: duplikasi dari book_id
    - work_id: integer
    - books_count: jumlah buku yang ada pada persediaan di **goodreads**
    - isbn: *Internatinal Standard Book Number* (integer)
    - isbn13: *International Standard Book Number* dengan 13 digit (integer)
    - authors: penulis buku (string)
    - original_publication_year: tahun publikasi buku (float)
    - original_title: judul buku (string)
    - title: judul pada situs **goodreads**
    - language_code: kode bahasa yang digunakan pada buku (string)
    - average rating: rata-rata dari rating buku (float)
    - ratings_count: jumlah rating yang diberikan oleh user untuk setiap buku (integer)
    - work_ratings_counts: jumlah rating (integer)
    - work_text_reviews_count: jumlah banyaknya review untuk setiap buku (integer)
    - rating_1: jumlah rating dengan nilai 1 (integer)
    - rating_2: jumlah rating dengan nilai 2 (integer)
    - rating_3: jumlah rating dengan nilai 3 (integer)
    - rating_4: jumlah rating dengan nilai 4 (integer)
    - rating_5: jumlah rating dengan nilai 5 (integer)
    - image_url: url dari cover buku (string)
    - small_image_url: url untuk cover buku dengan ukuran kecil (string)
- rating.csv
    - book_id: id dari buku pada situs **goodreads** (integer)
    - user_id: id dari setiap pelanggan (integer)
    - rating: rating yang diberikan oleh setiap user untuk setiap buku, dengan nilai antara 1 sampai dengan 5 (integer)

### Visualisasi Data
Melakukan *Exploratory Data Anlysis* menggunakan library pandas, yaitu berupa
- Melihat sturuktur data dari 2 data set dengan contoh data sebagai berikut:
    - books.csv
 
| id | book_id | best_book_id | work_id | books_count | authors                         | original_public<br>ation_year | original_title                              | title                                                          | ... |
|----|---------|--------------|---------|-------------|---------------------------------|-------------------------------|---------------------------------------------|----------------------------------------------------------------|-----|
| 1  | 2767052 | 2767052      | 2792775 | 272         | Suzanne Collins                 | 2008                          | The Hunger Games                            | The Hunger Games<br>(The Hunger Games, #1)                     | ... |
| 2  | 3       | 3            | 4640799 | 491         | J.K. Rowling,<br>Mary GrandPrÃ© | 1997                          | Harry Potter and<br>the Philosopher's Stone | Harry Potter and<br>the Sorcerer's Stone<br>(Harry Potter, #1) | ... |
| 3  | 41865   | 41865        | 3212258 | 226         | Stephenie Meyer                 | 2005                          | Twilight                                    | Twilight (Twilight, #1)                                        | ... |
| 4  | 2657    | 2657         | 3275794 | 487         | Harper Lee                      | 1960                          | To Kill a Mockingbird                       | To Kill a Mockingbird                                          | ... |
| 5  | 4671    | 4671         | 245494  | 1356        | F. Scott Fitzgerald             | 1925                          | The Great Gatsby                            | The Great Gatsby                                               | ... |

        
   - Informasi dari data set
    
RangeIndex: 10000 entries, 0 to 9999

| #  | Column                    | Non-Null Count | Dtype   |
|----|---------------------------|----------------|---------|
| 0  | id                        | 10000 non-null | int64   |
| 1  | book_id                   | 10000 non-null | int64   |
| 2  | best_book_id              | 10000 non-null | int64   |
| 3  | work_id                   | 10000 non-null | int64   |
| 4  | books_count               | 10000 non-null | int64   |
| 5  | isbn                      | 9300 non-null  | object  |
| 6  | isbn13                    | 9415 non-null  | float64 |
| 7  | authors                   | 10000 non-null | object  |
| 8  | original_publication_year | 9979 non-null  | float64 |
| 9  | original_title            | 9415 non-null  | object  |
| 10 | title                     | 10000 non-null | object  |
| 11 | language_code             | 8916 non-null  | object  |
| 12 | average_rating            | 10000 non-null | float64 |
| 13 | ratings_count             | 10000 non-null | int64   |
| 14 | work_ratings_count        | 10000 non-null | int64   |
| 15 | work_text_reviews_count   | 10000 non-null | int64   |
| 16 | ratings_1                 | 10000 non-null | int64   |
| 17 | ratings_2                 | 10000 non-null | int64   |
| 18 | ratings_3                 | 10000 non-null | int64   |
| 19 | ratings_4                 | 10000 non-null | int64   |
| 20 | ratings_5                 | 10000 non-null | int64   |
| 21 | image_url                 | 10000 non-null | object  |
| 22 | small_image_url           | 10000 non-null | object  |

dtypes: float64(3), int64(13), object(7)

memory usage: 1.8+ MB
        
   - Mencari nilai statsitik dari data set

|       |        Count |  Probability |
|------:|-------------:|-------------:|
| count | 1.472690e+05 | 1.472690e+05 |
|  mean | 2.481161e+03 | 6.790295e-06 |
|  std  | 4.645472e+04 | 1.271345e-04 |
|  min  | 1.000000e+00 | 2.736740e-09 |
|  25%  | 5.000000e+00 | 1.368370e-08 |
|  50%  | 1.700000e+01 | 4.652460e-08 |
|  75%  | 1.320000e+02 | 3.612500e-07 |
|  max  | 5.304407e+06 | 1.451679e-02 |

<br /><br />
   - rating.csv
 
|   | book_id | user_id | rating |
|--:|--------:|--------:|-------:|
| 0 |       1 |     314 |      5 |
| 1 |       1 |     439 |      3 |
| 2 |       1 |     588 |      5 |
| 3 |       1 |    1169 |      4 |
| 4 |       1 |    1185 |      4 |
        
   - Informasi dari data set
    
RangeIndex: 981756 entries, 0 to 981755

| #  | Column                    | Non-Null Count  | Dtype   |
|----|---------------------------|-----------------|---------|
| 0  | book_id                   | 981756 non-null | int64   |
| 1  | user_id                   | 981756 non-null | int64   |
| 2  | rating                    | 981756 non-null | int64   |

dtypes: int64(3)

memory usage: 1.8+ MB
        

<br /><br />
Melakukan visualisasi data menggunakan library matplotlib dan seaborn untuk memahami data lebih jauh, dengan visualisasi sebagai berikut:
- Top 10 dengan nilai rating tertinggi

|                           title                             |   cover book   |
|:-----------------------------------------------------------:|:--------------:|
|                The Complete Calvin and Hobbes               | ![top10-1](https://images.gr-assets.com/books/1473064526s/24812.jpg) |
|    Harry Potter Boxed Set, Books 1-5 (Harry Potter, #1-5)   | ![top10-2](https://s.gr-assets.com/assets/nophoto/book/50x75-a91bf249278a81aabab721ef782c4a74.png) |
|        Words of Radiance (The Stormlight Archive, #2)       | ![top10-3](https://images.gr-assets.com/books/1391535251s/17332218.jpg) |
|                   Mark of the Lion Trilogy                  | ![top10-4](https://images.gr-assets.com/books/1349032180s/95602.jpg) |
|                       ESV Study Bible                       | ![top10-5](https://images.gr-assets.com/books/1410151002s/5031805.jpg) |
|     It's a Magical World: A Calvin and Hobbes Collection    | ![top10-6](https://images.gr-assets.com/books/1437420710s/24814.jpg) |
| There's Treasure Everywhere: A Calvin and Hobbes Collection | ![top10-7](https://s.gr-assets.com/assets/nophoto/book/50x75-a91bf249278a81aabab721ef782c4a74.png) |
|           Harry Potter Boxset (Harry Potter, #1-7)          | ![top10-8](https://images.gr-assets.com/books/1392579059s/862041.jpg) |
|         Harry Potter Collection (Harry Potter, #1-6)        | ![top10-9](https://images.gr-assets.com/books/1328867351s/10.jpg) |
|             The Indispensable Calvin and Hobbes             | ![top10-10](https://s.gr-assets.com/assets/nophoto/book/50x75-a91bf249278a81aabab721ef782c4a74.png) |




- Top 10 buku terpopuler

|                           title                          |   cover book   |
|:--------------------------------------------------------:|:--------------:|
|          The Hunger Games (The Hunger Games, #1)         | ![pop10-1](https://images.gr-assets.com/books/1447303603s/2767052.jpg) |
| Harry Potter and the Sorcerer's Stone (Harry Potter, #1) | ![pop10-2](https://images.gr-assets.com/books/1474154022s/3.jpg) |
|                  Twilight (Twilight, #1)                 | ![pop10-3](https://images.gr-assets.com/books/1361039443s/41865.jpg) |
|                   To Kill a Mockingbird                  | ![pop10-4](https://images.gr-assets.com/books/1361975680s/2657.jpg) |
|                     The Great Gatsby                     | ![pop10-5](https://images.gr-assets.com/books/1490528560s/4671.jpg) |
|                  The Fault in Our Stars                  | ![pop10-6](https://images.gr-assets.com/books/1360206420s/11870085.jpg) |
|                        The Hobbit                        | ![pop10-7](https://images.gr-assets.com/books/1372847500s/5907.jpg) |
|                  The Catcher in the Rye                  | ![pop10-8](https://images.gr-assets.com/books/1398034300s/5107.jpg) |
|                    Pride and Prejudice                   | ![pop10-9](https://images.gr-assets.com/books/1320399351s/1885.jpg) |
|           Angels & Demons (Robert Langdon, #1)           | ![pop10-10](https://images.gr-assets.com/books/1303390735s/960.jpg) |


- Distribusi nilai rating oleh user yang diberikan kepada buku

![book-distribution](https://user-images.githubusercontent.com/105812169/204145974-c965c22c-abe5-48cf-835d-a64051aeb204.png)


- Penulis buku dengan rata-rata rating tertinggi

![author-rating](https://user-images.githubusercontent.com/105812169/204146061-32e77cec-2b6e-4d2b-a962-f3b637696991.png)


- Buku yang user ingin baca, yang masuk kedalam *wishlist* user

![wishlist-book](https://user-images.githubusercontent.com/105812169/204146124-67405cb1-870b-4b33-8e35-eda65accc74e.png)


Berdasarkan dari visual diatas dapat disimpulkan bahwa rating paling banyak berada di antara rating 3.5 dan 4, dan pada gambar kedua, Bill Waterson memiliki nilai rating yang sangat tinggi sebesar 4.82, dan buku yang ingin dibaca oleh user merupakan buku-buku yang sering kita dengar seperti Harry Potter dan Lord of the Rings.

<br /><br />

## Data Preparation

- **Content Based Model**

Tahapan yang dilakukan dalam penyelesaian masalah atau pembuatan sistem rekomendasi menggunakan bahasa pemrograman python, dan dilakukan beberapa tahap preparasi data, yaitu sebagai berikut:
    - Menghilangkan 849 duplikasi data pada books.csv
    - Membuat semua nilai di books.csv menjadi *lower case* dan menghilangkan N/A

|   |                      original_title |           average_rating | average_rating |
|--:|------------------------------------:|-------------------------:|---------------:|
| 0 |                      thehungergames |           suzannecollins |           4.34 |
| 1 | harrypotterandthephilosopher'sstone | j.k.rowling,marygrandpré |           4.44 |
| 3 |                  tokillamockingbird |                harperlee |           4.25 |
| 4 |                      thegreatgatsby |        f.scottfitzgerald |           3.89 |
| 5 |                  thefaultinourstars |                johngreen |           4.26 |

- Membuat bag of words yang akan digunakan sebagai fitur rekomendasi konten dengan menggabungkan variabel original_title, authors, dan average_rating menjadi satu *soup*, dengan contoh proses ini seperti tabel dibawah ini.

| index | original_title |                             authors |           average_rating | soup |                                                   |
|------:|---------------:|------------------------------------:|-------------------------:|-----:|---------------------------------------------------|
|   0   |              0 |                      thehungergames |           suzannecollins | 4.34 |                thehungergames suzannecollins 4.34 |
|   1   |              1 | harrypotterandthephilosopher'sstone | j.k.rowling,marygrandpré | 4.44 | harrypotterandthephilosopher'sstone j.k.rowlin... |
|   2   |              3 |                  tokillamockingbird |                harperlee | 4.25 |                 tokillamockingbird harperlee 4.25 |
|   3   |              4 |                      thegreatgatsby |        f.scottfitzgerald | 3.89 |             thegreatgatsby f.scottfitzgerald 3.89 |
|   4   |              5 |                  thefaultinourstars |                johngreen | 4.26 |                 thefaultinourstars johngreen 4.26 |

<br/><br/>

## Modeling and Result

- **Content Based Model**
Model yang digunakan menggunakan library scikit-learn dengan algoritma sebagai berikut:
    - TfidfVectorizer, Mengubah data text menjadi vector agar bisa dilakukan klasifikasi. Hal ini dilakukan untuk memudahkan dalam training, karena data yang tadinya string diubah menjadi bigram blocks dari karakter, dan merubahnya kedalam matrix.
    - cosine_similarity, Melakukan perhitungan derajat kesamaan *cosine similarity*, untuk mencari tahu derajat kesamaan dari setiap buku, dengan matrix 9151 x 9151, yang artinya mencari kesamaan antara 9151 buku. 
    - Membuat model rekomendasi dengan berdasarkan kesamaan yang dihitung dengan *cosine similarity*, dengan membuat fungsi dari *get_recommendation* dengan beberapa paramater sebagai berikut:
        - title: Judul buku.
        - cosine_sim: Dataframe mengenai kesamaan berdasarkan *cosine similarity*.
        - k: Banyaknya rekomendasi yang diberikan.
        - indicies: list dari fitur untuk rekomendasi berdasarkan kesamaan, dalam hal ini adalah 'original_title'.
    - Melakukan testing terhadap model rekomendasi yang dibuat, dengan input judul buku **The Hobbit**, dan menghasilkan Top 10 rekomendasi yaitu,
    
|    | Books Recommendation based on The Hobbit |
|---:|-----------------------------------------:|
|  1 |       The Hobbit or There and Back Again |
|  2 |                    The Lord of the Rings |
|  3 |               The Fellowship of the Ring |
|  4 |                           The Two Towers |
|  5 |     The Hobbit and The Lord of the Rings |
|  6 |                   The Return of the King |
|  7 |                        The Tommyknockers |
|  8 |                         The Tenth Circle |
|  9 |                                     Next |
| 10 |                    The Children of Húrin |

- **Collaborative Filtering Model**


## Evaluation
Pada model klasifikasi yang dibuat, menggunakan metric mean accuracy.
- Mean Accuracy merupakan nilai proporsi dari prediksi yang benar dari total prediksi.
- Rumus matematis Mean Accuracy adalah
    - Accuracy = Prediksi yang Benar / Total Prediksi.
- Hasil Accuracy dari beberapa algoritma yaitu sebagai berikut:
    - Multinomial Naive Bayes, Train Accuracy **72,72%**, Validation Accuracy **72.1%**
    - Logistic Regression, Train Accuracy **79,73%**, Validation Accuracy **79.19%**
    - Decission Tree, Train Accuracy **74,19%**, Validation Accuracy **72.24%**

- Evaluasi dari akurasi
    - Dari hasil tersebut dapat disimpulkan bahwa Logistic Regression sudah cocok untuk persoalan ini, karena kita akan melakukan klasifikasi dari 2 kategori yaitu Pria dan Wanita.
    - Nilai akurasi tidak terlalu tinggi dikarenakan daftar nama memiliki nilai yang ambigu jika jumlah Pria dan Wanita sama, jadi ia diklasifikasikan ke Wanita.
    - Lalu dari hasil prediksi pun ketika kita mencoba dengan nama yang populer dari keempat negara yang berbasis bahasa inggris yaitu AS, Kanada, UK dan Australia, maka hasil prediksi akan menghaslikan kesalahan, karena basis data hanya berasal dari keempat negara tersebut.

Dari hasil akurasi tersebut maka diplih Logistic Regression sebagai algoritma yang digunakan untuk klasifikasi dan dilakukan hyperparameters tuning, dan menghasilkan nilai parameter terbaik yaitu:
    
    penalty = 'l2'
    solver = 'lbfgs'
    
    
Parameter ini merupakan default parameter dari Logistic Regression menggunakan scikit-learn, parameter dpat ditambah saat melakukan tuning dengan beberapa parameter lain seperti pada penalty dapat menggunakan nilai 'none' dan 'elasticnet', serta pada parameter solver dapat juga ditambah dengan 'saga' dan 'sag'. Lalu dapat juga ditambah dengan parameter lain seperti C



## Referensi:
[Predicting customer’s gender and age depending on mobile phone data](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0180-9)
