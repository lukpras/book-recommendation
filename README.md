# Sistem Rekomendasi Buku menngunakan ***Content Based*** dan ***Collaborative Filtering*** - Luki Prasetyo

## **Project Overview**

Proyek ini merupakan sebuah proyek untuk membuat sistem rekomendasi berbasis kolaborasi dan berbasi konten. Membeli barang secara online telah menjadi tren saat ini, dibandingkan dengan pergi keluar dan membeli barang sendiri. Ketika membeli barang secara online maka, tedapat salah satu cara bagi penjual atau penyedia layanan belanja online untuk meningkatkan minat beli dari pembeli, salah satu caranya adalah dengan memberi rekomendasi barang, sehingga pembeli menjadi antusias, memudahkan pembeli untuk mencari barang lain dan memberikan cara yang lebih mudah dan cepat untuk membeli barang.

Karena di era saat ini yang kebanyakan kegiatan dapat dilakukan secara online, maka sistem rekomendasi dapat digunakan untuk meningkatkan kepuasan pelanggan dan juga meningkatkan aktivitas user, yang dapat berperan penting dalam membuat instansi atau perusahaan untuk meningkatkan kepuasan pelanggan dan juga meningkatkan pendapatan perusahaan, dengan menawarkan rekomendasi yang tepat bagi pelanggan/user. Sehingga sistem rekomendasi sangatlah penting, dimana pada kasus ini adalah rekomendasi untuk buku yang dibaca oleh pelanggan/user.

Permasalah tersebut dapat diatasi dengan melakukan pembuatan *machine learning* yang dapat melakukan rekomendasi berdasrkan beberapa fitur dari buku dengan melakukan pembuatan rekomendasi berdasrkan penulis buku dan rating buku menggunakan rekomendasi berbasi konten (*content based*) dan berbasis rating dari pelanggan/user menngunakan filter kolaborasi (*collaborative filtering*).

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
        
- Menghitung jumlah rating buku berdasrkan user.

    | rating | count  |
    |-------:|-------:|
    |    1   |  19485 |
    |    2   |  63010 |
    |    3   | 247698 |
    |    4   | 355878 |
    |    5   | 291198 |

- Mencari buku dengan nilai rating 0 atau tidak di rating.

    |     |     count | log_count |
    |----:|----------:|----------:|
    | 0.0 | 532822731 | 20.093699 |
    | 1.0 |     19485 |  9.877400 |
    | 2.0 |     63010 | 11.051049 |
    | 3.0 |    247698 | 12.419966 |
    | 4.0 |    355878 | 12.782343 |
    | 5.0 |    291198 | 12.581759 |

![book-rating-count](https://user-images.githubusercontent.com/105812169/204170154-61afb28e-bebe-4435-9379-75934b6bdca8.png)

- Menghitung banyaknya rating untuk setiap buku.

    | book_id | count |
    |--------:|------:|
    |    1    |  100  |
    |    2    |  100  |
    |    3    |  100  |
    |    4    |  100  |
    |    5    |  100  |

- Menghitung banyaknya rating yang diberikan oleh user

    | user_id | count |
    |--------:|------:|
    |    1    |   3   |
    |    2    |   3   |
    |    3    |   3   |
    |    4    |   3   |
    |    5    |   3   |

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


## Data Preparation

### **Content Based Model**

Tahapan yang dilakukan dalam penyelesaian masalah atau pembuatan sistem rekomendasi berbasis konten, dan dilakukan beberapa tahap preparasi data, yaitu sebagai berikut:

- Menghilangkan 849 duplikasi data pada books.csv
- Membuat semua nilai di books.csv menjadi *lower case* dan menghilangkan data yang tidak ada

    |   |                      original_title |           average_rating | average_rating |
    |--:|------------------------------------:|-------------------------:|---------------:|
    | 0 |                      thehungergames |           suzannecollins |           4.34 |
    | 1 | harrypotterandthephilosopher'sstone | j.k.rowling,marygrandpré |           4.44 |
    | 3 |                  tokillamockingbird |                harperlee |           4.25 |
    | 4 |                      thegreatgatsby |        f.scottfitzgerald |           3.89 |
    | 5 |                  thefaultinourstars |                johngreen |           4.26 |

- Membuat bag of words yang akan digunakan sebagai fitur rekomendasi konten dengan menggabungkan variabel original_title, authors, dan average_rating menjadi satu *soup*, dengan hasil seperti tabel dibawah ini. Hal ini dilakukan karena fitur genre tidak diketahui maka perlu dibaut fitur yang dapat membuat content based filtering menjadi tepat.

    |       | index |                  original_title |           authors | average_rating | soup                                                   |
    |------:|---------------:|------------------------------------:|-------------------------:|-----:|---------------------------------------------------|
    |   0   |              0 |                      thehungergames |           suzannecollins | 4.34 |                thehungergames suzannecollins 4.34 |
    |   1   |              1 | harrypotterandthephilosopher'sstone | j.k.rowling,marygrandpré | 4.44 | harrypotterandthephilosopher'sstone j.k.rowlin... |
    |   2   |              3 |                  tokillamockingbird |                harperlee | 4.25 |                 tokillamockingbird harperlee 4.25 |
    |   3   |              4 |                      thegreatgatsby |        f.scottfitzgerald | 3.89 |             thegreatgatsby f.scottfitzgerald 3.89 |
    |   4   |              5 |                  thefaultinourstars |                johngreen | 4.26 |                 thefaultinourstars johngreen 4.26 |


### **Collaborative Filtering**

Tahapan yang dilakukan dalam penyelesaian masalah atau pembuatan sistem rekomendasi berbasis kolaborasi user, dan dilakukan beberapa tahap preparasi data, yaitu sebagai berikut:
    
- Menghilangkan null values dari book_id dan original_title.
- Menghilangkan buku yang tidak di rating atau memilki rating 0, dengan mengambil buku yang telah dilakukan rating sebanyak 50 kali, hasil ini dilakukan karena untuk mencari tahu reaksi antara user terhadap buku-buku.
    
    |     |     count | log_count |
    |----:|----------:|----------:|
    | 0.0 | 532822731 | 20.093699 |
    | 1.0 |     19485 |  9.877400 |
    | 2.0 |     63010 | 11.051049 |
    | 3.0 |    247698 | 12.419966 |
    | 4.0 |    355878 | 12.782343 |
    | 5.0 |    291198 | 12.581759 |
   
- Menghilangan user yang melakukan pemberian rating dibawah 50 kali, untuk mencari tahu interaksi antara user terhadap buku-buku.
- Membuat dataframe baru berdasarkan buku yang telah dihilangkan rating 0 dan user yang telah melakukan rating diatas 50 kali.
 
    |       | book_id | user_id | rating | original_title                         |
    |------:|--------:|--------:|-------:|----------------------------------------|
    |   0   |       1 |     314 |      5 | Harry Potter and the Half-Blood Prince |
    |   1   |       1 |     439 |      3 | Harry Potter and the Half-Blood Prince |
    |   2   |       1 |     588 |      5 | Harry Potter and the Half-Blood Prince |
    |   3   |       1 |    1169 |      4 | Harry Potter and the Half-Blood Prince |
    |   4   |       1 |    1185 |      4 | Harry Potter and the Half-Blood Prince |
    |  ...  |     ... |     ... |    ... |                                    ... |
    | 33258 |    9998 |    8078 |      5 |                  砂の女 [Suna no onna] |
    | 33259 |    9998 |   14122 |      4 |                  砂の女 [Suna no onna] |
    | 33260 |    9998 |   25988 |      5 |                  砂の女 [Suna no onna] |
    | 33261 |    9998 |   31162 |      3 |                  砂の女 [Suna no onna] |
    | 33262 |    9998 |   52330 |      4 |                   砂の女 [Suna no onna |

33263 rows × 4 columns

- Melakukan random data pada dataframe yang akan dilkaukan pemisahan menjadi train dan validation.
- Melakukan pemisahan data menjadi train dan validation data, dengan ukuran training sebsar 80% dari data set


## Modeling and Result
### **Content Based Model**
Model ini merupakan, model yang digunakan untuk membuat sistem rekomendasi yang akan melakukan penebakan dari user, berdasarkan aktivitas dari user, dengan menggunakan keywords dan atribut yang di sematkan pada objek, contohnya pada kasus ini adalah judul buku, author, dan rata-rata rating, dan melakukan pencocokan terhadap profil dari user.

Adapun kelebihan dan kekurangan Content Based Model adalah sebagai berikut:
- Kelebihan
    - Model ini memiliki kelebihan karena tidak perlunya data dari user untuk membuat sistem rekomendasi.
    - Kemungkinan relevansi terhadap keinginan user cukup tinggi, karena rekomendasi dapat sesuai dengan keinginan user.
    - Mudah dalam pembuatan.
- Kekurangan
    - Model hanya dapat membuat rekomendasi berbasis ketertarikan user, sehingga model akan kaku jiga ingin membuat rekomendasi diluar ketertarikan user.

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


### **Collaborative Filtering Model**
Collaborative filtering bergantung pada komuntias dari user. Ia tidak memerlukan atribut untuk setiap itemnya seperti pada sistem berbasis konten. Collaborative filtering. Collaborative filtering berdasarkan dari interaksi masa lalu antara user dan item (buku), sehingga user akan diberi rekomendasi dari user-user yang serupa.

Adapun kelebihan dan kekurangan Collaborative filtering Model adalah sebagai berikut:
- Kelebihan
    - Tidak membutuhkan domain knowledge dari atribut yang ada.
    - Model dapat membuat user menemupakn minat baru, karena berdasarkan history dari user-user lain.
- Kekurangan
    - TIdak dapat melakukan rekomendasi dari item (buku) yang baru, karena belum terjadi interaksi antara item dan user.
- Model yang digunakan menggunakan class model pada library Keras dan, dengan algoritma yang digunakan adalah RecommenderNet.
    - RecommenderNet bekerja dengan melakukan embedding user dan buku kedala vektor 50 dimensi, model ini menghitung skor kecocokan antara embed buku melalui titik, dan menambahkan bias untuk setiap buku dan setiap user, skor kesamaan memiliki skala antara 0 dan 1 dengan sigmoid.
- Compile Model, dengan menggunakan parameter:
    - loss, dengan BinaryCrossentropy
    - optimizer, menggunakan Adam Optimizer dengan learning rate 0.001
    - metrics, menggunakan Root Mean Square Error (RMSE)
- Training model dengan 100 iterasi menghasilkan

```
Epoch 1/100
3327/3327 [==============================] - 15s 5ms/step - loss: 0.6078 - root_mean_squared_error: 0.2552 - val_loss: 0.5959 - val_root_mean_squared_error: 0.2458
Epoch 2/100
3327/3327 [==============================] - 20s 6ms/step - loss: 0.5825 - root_mean_squared_error: 0.2326 - val_loss: 0.5994 - val_root_mean_squared_error: 0.2490
Epoch 3/100
3327/3327 [==============================] - 21s 6ms/step - loss: 0.5736 - root_mean_squared_error: 0.2243 - val_loss: 0.5921 - val_root_mean_squared_error: 0.2425
...
Epoch 98/100
3327/3327 [==============================] - 15s 4ms/step - loss: 0.5207 - root_mean_squared_error: 0.1792 - val_loss: 0.6042 - val_root_mean_squared_error: 0.2315
Epoch 99/100
3327/3327 [==============================] - 15s 4ms/step - loss: 0.5211 - root_mean_squared_error: 0.1793 - val_loss: 0.6044 - val_root_mean_squared_error: 0.2316
Epoch 100/100
3327/3327 [==============================] - 16s 5ms/step - loss: 0.5206 - root_mean_squared_error: 0.1791 - val_loss: 0.6051 - val_root_mean_squared_error: 0.2318
```

- Melakukan testing terhadap model collaborative filtering yang telah dibuat, dengan hasil sebagai berikut:
    - Mengambil user random dan memperlihatkan rekomendasi untuk user tersebut berdasarakan buku yang ia baca
        ```
        Showing recommendations for users: 32592

        ================================
        Book with high ratings from user
        --------------------------------
        Harry Potter and the Philosopher's Stone
        The Hitchhiker's Guide to the Galaxy
        Heidi
        The Lost Continent: Travels in Small-Town America
        Zodiac
    
        ================================
        Top 10 book recommendation
        --------------------------------
        Fahrenheit 451
        American Gods
        Still Life with Woodpecker
        Next
        Rachel's Holiday
        Little Town on the Prairie
        Lucy Sullivan is Getting Married
        Bokhandleren i Kabul
        Four Blondes
        Motor Mouth
        ```
    - Melakukan rekomendasi berdasarkan buku yang dipilih
        ```
        ================================
        Book you read
        --------------------------------
        Peter and the Shadow Thieves

        ================================
        Top 10 book recommendation
        --------------------------------
        Atonement
        A People's History of the United States: 1492 to Present 
        Still Life with Woodpecker
        Next
        Rachel's Holiday
        Lucy Sullivan is Getting Married
        Bokhandleren i Kabul
        Amsterdam
        The Taste of Home Cookbook
        The Android's Dream

## Evaluation
### Content Based Model
Metrik yang digunakan pada model ini adalah precision dimana rumus precision adalah:
- Precision = Rekomendasi yang relevan / Rekomendasi yang diberikan
- Dengan hasil precision sebesar 70%
- Precison diambil dari relevansi antara authors dan rating yang sesuai dengan buku yang dipilih

#### Evaluasi Pertama
- Buku yang dicari rekomedasinya

    |   |              original_title |                                                authors | average_rating |
    |--:|----------------------------:|-------------------------------------------------------:|---------------:|
    | 0 |   The Hobbit                | Chuck Dixon, J.R.R. Tolkien, David Wenzel, Sean Deming |           4.25 |

- Rekomedasi

    |   |                       original_title |                                       authors | average_rating |
    |--:|-------------------------------------:|----------------------------------------------:|---------------:|
    | 0 |   The Hobbit or There and Back Again |                                J.R.R. Tolkien |           4.25 |
    | 1 |                The Lord of the Rings |                                J.R.R. Tolkien |           4.47 |
    | 2 |           The Fellowship of the Ring |                                J.R.R. Tolkien |           4.34 |
    | 3 |                       The Two Towers |                                J.R.R. Tolkien |           4.42 |
    | 4 | The Hobbit and The Lord of the Rings |                                J.R.R. Tolkien |           4.59 |
    | 5 |               The Return of the King |                                J.R.R. Tolkien |           4.51 |
    | 6 |                    The Tommyknockers |                                  Stephen King |           3.48 |
    | 7 |                     The Tenth Circle |                                  Jodi Picoult |           3.48 |
    | 8 |                                 Next |                              Michael Crichton |           3.48 |
    | 9 |                The Children of Húrin | J.R.R. Tolkien, Christopher Tolkien, Alan Lee |           3.94 |

Pada evaluasi pertama terdapat ketidaksesuaian dari judul buku dan authors, karena kita tidak memiliki genre sebagai fitur untuk melakukan rekomendasi maka, rekomendasi yang diberikan kemungkinan belum tepat presisisnya, dan pada buku yang dicari rekomendasinya memiliki 5 author yang akan berpengaruh untuk hasil rekomendasi.

Karena kita harus tahu terlebih dahulu konteks dalam buku tersebut untuk menilai apakah rekomendasi sudah benar atau tidak, dan perlu pengetahuan dari orang yang memiliki pengetahuan antara buku hasil rekomendasi diatas sudah presisi atau belum.

Jika kita mengambil authors saja sebagai nilai ketepatan bagi perhitungan preisisi ini maka nilai preisi sebagai berikut:

```
Recommender system precision = recommendation that relevant / items we recommendend
Recommender system precision = 7 / 10
Recommender system precision = 70%
```

Presisi yang dihaislkan dari rekomendasi sebsar 70%

#### Evaluasi Kedua
- Buku yang dicari rekomedasinya

    |   |              original_title |          authors | average_rating |
    |--:|----------------------------:|-----------------:|---------------:|
    | 0 |   Jurassic Park             | Michael Crichton |           3.96 |

- Rekomedasi

    |   |          original_title |          authors | average_rating |
    |--:|------------------------:|-----------------:|---------------:|
    | 0 |          The Lost World | Michael Crichton |           3.72 |
    | 1 |    The Andromeda Strain | Michael Crichton |           3.87 |
    | 2 |                Timeline | Michael Crichton |           3.83 |
    | 3 | The Great Train Robbery | Michael Crichton |           3.84 |
    | 4 |                  Sphere | Michael Crichton |           3.77 |
    | 5 |        The Terminal Man | Michael Crichton |           3.34 |
    | 6 |              Disclosure | Michael Crichton |           3.76 |
    | 7 |                    Prey | Michael Crichton |           3.72 |
    | 8 |           State of Fear | Michael Crichton |           3.69 |
    | 9 |                Airframe | Michael Crichton |           3.66 |

Pada evalusi kedua menghasilkan presisi yang baik karena author dari buku yang dicari hanya satu dan pada database author tersebut memiliki lebih dari 10 buku, sehingga rekomendasi akan diberikan berdasarkan author tersebut dengan nilai presisi sebesar 100%


## Collaborative Filtering

Pada Collaborative filtering, menggunakan metrics Root Mean Square Error (RMSE).

Root Mean Square Error (RMSE) adalah standar deviasi dari residual (kesalahan prediksi). Residu adalah ukuran seberapa jauh dari titik data garis regresi; RMSE adalah ukuran seberapa tersebar residu tersebut. Dengan kata lain, seberapa terkonsentrasinya data di sekitar garis yang paling sesuai.
RMSE baik digunakan untuk melakukan prediksi terhadap nilai numerik.

Rumus untuk menghitung RMSE adalah sebagai berikut:

![RMSE](https://user-images.githubusercontent.com/105812169/204192286-1b6ee340-e212-48b5-9120-91491ddeb1ec.jpg)

Dimana,
- *predicted i* = Nilai prediksi dari itkerasi ke-i
- *actual i* = Nilai aktual yang di observasi pada iterasi ke-i
- N = Total jumlah dari observasi

Perbedaan antara prediksi dan aktual adalah residual.

Dan pada gambar dibawah ini merupakan nilai RMSE dari iterasi ke-1 hingga iterasi ke-100

![collaborative-metrics](https://user-images.githubusercontent.com/105812169/204176964-211f853d-b9e6-4345-9958-1cdfaa1e9458.png)

Pada gambar diatas dapat dilihat bahwa nilai RMSE untuk set training menyentuh titik *steady state* pada iterasi ke 45 dan untuk set validasi menyentuh titik *steady state* pada iterasi ke 20, diatas iterasi tersebut, nilai RMSE tidak berubah secara signifikan.

Dengan nilai RMSE pada iterasi ke terakhir sebesar 0.17 untuk set training dan 0.23 untuk set validasi, dimana nilai ini cukup bagus karena berada di sekitar 0.1 sampai dengan 0.5. Sehingga model yang kita buat dapat melakukan rekomendasi.





## Referensi:
- [*T. Anwar, V. Uma and Shahjad, "Book Recommendation for eLearning Using Collaborative Filtering and Sequential Pattern Mining," 2020 International Conference on Data Analytics for Business and Industry: Way Towards a Sustainable Economy (ICDABI), 2020*](https://ieeexplore.ieee.org/document/9325599)
- [*P. Mathew, B. Kuriakose and V. Hegde, "Book Recommendation System through content based and collaborative filtering method," 2016 International Conference on Data Mining and Advanced Computing (SAPIENCE), 2016*](https://ieeexplore.ieee.org/document/7684166)
