import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
import pickle
with st.sidebar:
    selected=option_menu('Prediction Emission in Rwanda',['Dashboard','Start Prediction','About Us'])
st.title('PREDIKSI EMISI DI RWANDA')
st.write('Selamat datang, di web prediksi emisi di wilayah Rwanda.')
if selected=='Dashboard':
    image_path = "co2.jpg"
    st.image(image_path, caption="emission", use_column_width=True, output_format="auto")

    
    st.write('Tujuan dari penelitian ini sebagai berikut :')
    st.markdown('''
    <b>1. Memproyeksikan Emisi Karbon di Rwanda pada Tahun 2022 </b> 
    <br> Menggunakan data historis dan model prediksi untuk memperkirakan jumlah emisi karbon yang akan dihasilkan oleh berbagai sektor di Rwanda pada tahun 2022 dan pada tahun ke depannya.
    <br>

    <b>2. Menganalisis Faktor Penyebab Emisi </b>
    <br>
    Mengidentifikasi faktor-faktor utama yang berkontribusi terhadap emisi karbon di Rwanda, termasuk sektor transportasi, industri, pertanian, dan penggunaan lahan.
    <br>
    <br>
    <b>3. Menilai Dampak Lingkungan dan Sosial</b>
    <br>
    Mengevaluasi dampak dari emisi karbon terhadap lingkungan dan masyarakat di Rwanda, termasuk dampak terhadap kesehatan, pertanian, dan ekosistem.
    <br>
    <br>
    <b>4. Menyusun Strategi Pengurangan Emisi</b> 
    <br>
    Mengembangkan rekomendasi kebijakan dan strategi untuk mengurangi emisi karbon di Rwanda, dengan fokus pada solusi yang berkelanjutan dan dapat diterapkan.</b>
   
''', unsafe_allow_html=True)


if selected=='Start Prediction':
   
    # Mengunggah file
    uploaded_file = st.file_uploader("Unggah file CSV di sini", type=['csv'], accept_multiple_files=False)
    # Memeriksa apakah file diunggah
    if uploaded_file is not None:
        # Membaca file yang diunggah
        df_file = pd.read_csv(uploaded_file)
        # Menampilkan contoh 5 data dari file yang diunggah
        st.write('Contoh 5 data yang ditampilkan:')
        st.write(df_file.head(5))
    else:
        st.write('Mohon unggah file dalam format CSV')
    button = st.button('PREDIKSI', use_container_width=1000, type='primary')
    if button:
        # Memeriksa apakah file diunggah dan dataframe telah dibuat
        if uploaded_file is not None:
            # Proses prediksi
            # List kolom numerik
            id_column=df_file['ID_LAT_LON_YEAR_WEEK']
            
            numeric_columns = df_file.select_dtypes(include='number').columns
            # Mengisi nilai yang hilang pada kolom numerik dengan mean
            for col in numeric_columns:
                df_file[col].fillna(df_file[col].mean(), inplace=True)
            
            df_clean= df_file.drop(columns=['ID_LAT_LON_YEAR_WEEK'])
          
            with open('scaler.pkl', 'rb') as file:
                normalisasi = pickle.load(file)
            norm_data = normalisasi.transform(df_clean)
            
           
            with open('LGBM.pkl', 'rb') as file:
                load_model = pickle.load(file)
            predictions = load_model.predict(norm_data)
            # Membuat DataFrame dari hasil prediksi
            df_pred = pd.DataFrame({'Predicted': predictions})
            # Menambahkan kembali kolom 'id' ke DataFrame hasil prediksi
            df_pred_id = pd.concat([id_column.reset_index(drop=True), df_pred], axis=1)
            #Tampilkan DataFrame yang telah digabungkan kembali
            st.write(df_pred_id)


        else:
            st.write('Mohon unggah file CSV terlebih dahulu')

if selected=='About Us':
    st.write('kelompok 2 - Data Science SIB cycle 6 GreatEdu')
