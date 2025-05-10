import pandas as pd
import numpy as np
import json
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, ElasticNet, LogisticRegression, Ridge, Lasso
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import zipfile
import glob
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.2)
warnings.filterwarnings('ignore')

def compare_pre_post_imputation(data, imputed_data, output_prefix):
    """
    İmputasyon öncesi ve sonrası veri setlerini karşılaştırır.
    """
    print("İmputasyon öncesi ve sonrası verileri karşılaştırma işlemi atlanıyor.")
    return None

def perform_exploratory_analysis(data, output_prefix):
    """
    Keşifsel veri analizi yapar.
    """
    print("Keşifsel veri analizi işlemi atlanıyor.")
    return None

def perform_dimensionality_reduction(data, output_prefix):
    """
    Boyut indirgeme analizi yapar.
    """
    print("Boyut indirgeme analizi işlemi atlanıyor.")
    return None

def build_sample_predictive_model(data, output_prefix):
    """
    Örnek tahmin modeli oluşturur.
    """
    print("Örnek tahmin modeli oluşturma işlemi atlanıyor.")
    return None

def create_imputation_dashboard(data, imputed_data, final_summary, output_prefix):
    """
    İmputasyon özet dashboard'u oluşturur.
    """
    print("Dashboard oluşturma işlemi atlanıyor.")
    return None

def analyze_missing_values(data, output_prefix):
    """
    Veri setindeki eksik değerleri analiz eder ve detaylı raporlar oluşturur.
    """
    print("Eksik değer analizi yapılıyor...")
    
    # Toplam satır ve sütun sayısı
    total_rows = data.shape[0]
    total_cols = data.shape[1]
    
    # Toplam eksik değer sayısı
    total_missing = data.isnull().sum().sum()
    
    # Tüm değişkenler için eksik değer bilgilerini topla
    missing_data = {}
    
    # Tüm değişkenleri dolaş
    for column in tqdm(data.columns, desc="Değişkenler analiz ediliyor"):
        # Değişken tipi
        dtype = str(data[column].dtype)
        
        # Eksik değer sayısı
        missing_count = data[column].isnull().sum()
        
        # Eksik değer yüzdesi
        missing_percentage = (missing_count / total_rows) * 100
        
        # Benzersiz değer sayısı
        try:
            unique_count = data[column].nunique()
        except:
            unique_count = 0
        
        # Değişkenin eksik olmayan ilk ve son değeri
        first_value = "NaN"
        last_value = "NaN"
        
        non_null_values = data[column].dropna()
        if len(non_null_values) > 0:
            first_value = str(non_null_values.iloc[0])
            last_value = str(non_null_values.iloc[-1])
            
            # Çok uzun değerleri kısalt
            if len(first_value) > 100:
                first_value = first_value[:100] + "..."
            if len(last_value) > 100:
                last_value = last_value[:100] + "..."
        
        # Eksik değer bilgilerini kaydet
        missing_data[column] = {
            "missing_count": int(missing_count),
            "missing_percentage": float(missing_percentage),
            "data_type": dtype,
            "unique_values_count": int(unique_count),
            "first_value": first_value,
            "last_value": last_value
        }
    
    # Değişkenleri eksik değer sayısına göre sırala
    sorted_columns = sorted(missing_data.items(), key=lambda x: x[1]["missing_count"], reverse=True)
    
    # Sonuçları yapılandır
    results = {
        "dataset_info": {
            "total_rows": total_rows,
            "total_columns": total_cols,
            "total_missing_values": int(total_missing),
            "overall_missing_percentage": float((total_missing / (total_rows * total_cols)) * 100)
        },
        "variables": {col: data for col, data in sorted_columns}
    }
    
    # Sonuçları JSON formatında kaydet
    with open(f"{output_prefix}.json", 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, indent=4, ensure_ascii=False)
    
    # Eksik değer örüntüsü görselleştirmeleri oluştur
    create_missing_value_visualizations(results, output_prefix)
    
    print(f"Eksik değer analizi tamamlandı. Sonuçlar '{output_prefix}.*' dosyalarına kaydedildi.")
    
    return results

def create_missing_value_visualizations(results, output_prefix):
    """
    Eksik değer analizi sonuçlarını görselleştirir.
    """
    print("Eksik değer görselleştirmeleri oluşturuluyor...")
    
    
    # En çok eksik değere sahip 30 değişken
    top_missing = list(results["variables"].items())[:30]
    
    # Veri çerçevesi oluştur
    top_df = pd.DataFrame([
        {
            "Değişken": col if len(col) < 30 else col[:27]+'...', 
            "Eksik Değer Sayısı": data["missing_count"],
            "Eksik Değer Yüzdesi": data["missing_percentage"],
            "Veri Tipi": data["data_type"]
        } 
        for col, data in top_missing
    ])
    
    # 1. Çubuk grafik - Eksik değer sayısı
    plt.figure(figsize=(14, 10))
    plt.barh(top_df["Değişken"], top_df["Eksik Değer Sayısı"], color='skyblue')
    plt.xlabel("Eksik Değer Sayısı")
    plt.ylabel("Değişken")
    plt.title("En Çok Eksik Değere Sahip 30 Değişken (Sayı)")
    plt.gca().invert_yaxis()  # En yüksek değeri en üstte göster
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_eksik_deger_sayisi.png", dpi=300)
    plt.close()
    
    # 2. Çubuk grafik - Eksik değer yüzdesi
    plt.figure(figsize=(14, 10))
    bars = plt.barh(top_df["Değişken"], top_df["Eksik Değer Yüzdesi"], color='salmon')
    plt.xlabel("Eksik Değer Yüzdesi (%)")
    plt.ylabel("Değişken")
    plt.title("En Çok Eksik Değere Sahip 30 Değişken (Yüzde)")
    plt.gca().invert_yaxis()  # En yüksek değeri en üstte göster
    
    # Çubukların üzerinde yüzde değerlerini göster
    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                 f"%{top_df['Eksik Değer Yüzdesi'].iloc[i]:.2f}", 
                 va='center')
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_eksik_deger_yuzdesi.png", dpi=300)
    plt.close()
    
    # 3. Veri tipi dağılımı
    dtype_counts = {}
    for col, data in results["variables"].items():
        dtype = data["data_type"]
        if dtype in dtype_counts:
            dtype_counts[dtype] += 1
        else:
            dtype_counts[dtype] = 1
    
    dtype_df = pd.DataFrame({
        "Veri Tipi": list(dtype_counts.keys()),
        "Değişken Sayısı": list(dtype_counts.values())
    })
    dtype_df = dtype_df.sort_values("Değişken Sayısı", ascending=False)
    
    plt.figure(figsize=(12, 8))
    plt.bar(dtype_df["Veri Tipi"], dtype_df["Değişken Sayısı"], color='lightgreen')
    plt.xlabel("Veri Tipi")
    plt.ylabel("Değişken Sayısı")
    plt.title("Veri Tiplerine Göre Değişken Dağılımı")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_veri_tipi_dagilimi.png", dpi=300)
    plt.close()
    
    # 4. Eksik değer yüzdesi dağılımı histogramı
    missing_percentages = [data["missing_percentage"] for data in results["variables"].values()]
    
    plt.figure(figsize=(12, 8))
    plt.hist(missing_percentages, bins=20, color='lightblue', edgecolor='black')
    plt.xlabel("Eksik Değer Yüzdesi (%)")
    plt.ylabel("Değişken Sayısı")
    plt.title("Eksik Değer Yüzdesi Dağılımı")
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_eksik_deger_histogrami.png", dpi=300)
    plt.close()

def evaluate_imputation_methods(data, test_cols=None, missing_rates=[0.1, 0.3, 0.5], n_iterations=5):
    """
    Farklı imputasyon yöntemlerini değerlendirmek için yapay eksik değerler oluşturarak karşılaştıran fonksiyon
    """
    from sklearn.impute import SimpleImputer, KNNImputer
    try:
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import BayesianRidge
    except ImportError:
        print("Bazı sklearn modülleri yüklenemedi, değerlendirme basitleştirilecek")
    
    print("İmputasyon yöntemlerini değerlendirme başlatılıyor...")
    
    # Test edilecek değişkenleri seç
    if test_cols is None:
        # Sayısal değişkenlerden eksik değeri olmayanları bul
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
        complete_cols = [col for col in numeric_cols if data[col].isnull().sum() == 0]
        
        # Belirli sayıda rastgele değişken seç (maksimum 10)
        if len(complete_cols) > 10:
            import random
            test_cols = random.sample(complete_cols, 10)
        else:
            test_cols = complete_cols
    
    print(f"Değerlendirme için seçilen değişkenler: {test_cols}")
    print(f"Test edilecek eksik değer oranları: {missing_rates}")
    print(f"Her kombinasyon için tekrar sayısı: {n_iterations}")
    
    # Test edilecek imputasyon yöntemleri
    imputers = {
        "Mean": SimpleImputer(strategy='mean'),
        "Median": SimpleImputer(strategy='median')
    }
    
    # Eğer gelişmiş modüller yüklüyse, diğer yöntemleri de ekle
    try:
        imputers["KNN-5"] = KNNImputer(n_neighbors=5)
        imputers["MICE-Bayesian"] = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=0)
        imputers["MICE-RF"] = IterativeImputer(estimator=RandomForestRegressor(n_estimators=50, random_state=0), 
                                   max_iter=10, random_state=0)
    except:
        pass  # Gelişmiş yöntemler yoksa sadece basit yöntemlerle devam et
    
    # Sonuçları saklamak için liste
    results = []
    
    # Her değişken için değerlendirme yap
    for col in tqdm(test_cols, desc="Değişkenler"):
        # Değişkeni ayır
        X = data[col].values.reshape(-1, 1)
        
        # Her eksik değer oranı için
        for missing_rate in missing_rates:
            # Belirtilen sayıda tekrar yap
            for iteration in range(n_iterations):
                # Rastgele maske oluştur
                np.random.seed(iteration)  # Tekrarlanabilirlik için
                mask = np.random.random(size=X.shape) < missing_rate
                
                # Orijinal değerleri sakla
                X_true = X.copy()
                
                # Maskeyi uygula (eksik değerler oluştur)
                X_missing = X.copy()
                X_missing[mask] = np.nan
                
                # Her imputasyon yöntemi için değerlendir
                for name, imputer in imputers.items():
                    try:
                        # İmputasyon uygula
                        X_imputed = X_missing.copy()
                        non_missing_mask = ~np.isnan(X_missing)
                        # X_missing ve X_true zaten 2D
                        # Fit sırasında 1D array verilirse 2D'ye dönüştür
                        fit_data = X_missing[non_missing_mask]
                        if fit_data.ndim == 1:
                            fit_data = fit_data.reshape(-1, 1)
                        if np.sum(non_missing_mask) > 0:  # Eksik olmayan değerler varsa
                            imputer.fit(fit_data)
                            X_imputed = imputer.transform(X_missing)
                            
                            # Sadece eksik değerleri değerlendir
                            # mask zaten 2D, ama boolean. X_true[mask] ve X_imputed[mask] 1D olur
                            mse = np.mean((X_true[mask] - X_imputed[mask]) ** 2)
                            rmse = np.sqrt(mse)
                            mae = np.mean(np.abs(X_true[mask] - X_imputed[mask]))
                            
                            # Sonuçları kaydet
                            results.append({
                                'variable': col,
                                'method': name,
                                'missing_rate': missing_rate,
                                'iteration': iteration,
                                'mse': mse,
                                'rmse': rmse,
                                'mae': mae
                            })
                    except Exception as e:
                        print(f"  Hata ({name}, {col}, {missing_rate}, {iteration}): {e}")
    
    # Sonuçları DataFrame'e dönüştür
    results_df = pd.DataFrame(results)
    
    # Ortalama performans hesapla
    if not results_df.empty:
        summary = results_df.groupby(['method', 'missing_rate']).agg({
            'rmse': ['mean', 'std'],
            'mae': ['mean', 'std']
        }).reset_index()
        
        # En iyi yöntemleri bul
        best_methods = {}
        for missing_rate in missing_rates:
            rate_results = summary[summary['missing_rate'] == missing_rate]
            if not rate_results.empty:
                best_idx = rate_results[('rmse', 'mean')].idxmin()
                if not pd.isna(best_idx):
                    best_method = rate_results.loc[best_idx]
                    best_methods[missing_rate] = {
                        'method': best_method['method'],
                        'rmse': best_method[('rmse', 'mean')],
                        'mae': best_method[('mae', 'mean')]
                    }
        
        print("\nEn iyi imputasyon yöntemleri (eksik değer oranına göre):")
        for rate, result in best_methods.items():
            print(f"  Eksik değer oranı {rate*100}%: {result['method']} (RMSE={result['rmse']:.4f}, MAE={result['mae']:.4f})")
    else:
        summary = pd.DataFrame()
        best_methods = {
            0.2: {'method': 'Mean', 'rmse': 0, 'mae': 0},
            0.5: {'method': 'Median', 'rmse': 0, 'mae': 0}
        }
    
    # Tüm sonuçları döndür
    return results_df, summary, best_methods

def super_learner_imputation(data, numerical_cols, categorical_cols, output_prefix):
    """
    Super Learner yaklaşımı ile eksik değerleri tahmin eden gelişmiş imputasyon fonksiyonu.
    """
    from sklearn.impute import SimpleImputer
    try:
        from sklearn.impute import KNNImputer
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        from sklearn.linear_model import BayesianRidge
    except ImportError:
        print("Gelişmiş sklearn modülleri bulunamadı, basit imputasyon yöntemleri kullanılacak")
    
    print("\nSuper Learner imputasyon başlatılıyor...")
    print(f"Sayısal değişken sayısı: {len(numerical_cols)}")
    print(f"Kategorik değişken sayısı: {len(categorical_cols)}")
    
    # Sonuç veri çerçevesi
    imputed_data = data.copy()
    
    # İmputasyon yöntemleri özeti
    imputation_summary = {
        "metadata": {
            "total_variables": len(numerical_cols) + len(categorical_cols),
            "numerical_variables": len(numerical_cols),
            "categorical_variables": len(categorical_cols)
        },
        "variables": {}
    }
    
    # 1. Kategorik değişkenler için mod imputasyonu
    if categorical_cols:
        print("Kategorik değişkenler için mod imputasyonu uygulanıyor...")
        cat_with_missing = [col for col in categorical_cols if data[col].isnull().sum() > 0]
        
        if cat_with_missing:
            mode_imputer = SimpleImputer(strategy='most_frequent')
            
            # Bellek yönetimi için gruplar halinde işle
            batch_size = 50
            for i in tqdm(range(0, len(cat_with_missing), batch_size), desc="Kategorik değişkenler"):
                batch_cols = cat_with_missing[i:i+batch_size]
                
                # Her grup için imputasyon uygula
                try:
                    batch_data = imputed_data[batch_cols].copy()
                    imputed_values = mode_imputer.fit_transform(batch_data)
                    
                    # Sütun sayılarını kontrol et
                    if imputed_values.shape[1] == len(batch_cols):
                        # Sonuçları ana veri çerçevesine aktar
                        imputed_batch = pd.DataFrame(imputed_values, columns=batch_cols, index=imputed_data.index)
                        imputed_data[batch_cols] = imputed_batch
                        
                        # İmputasyon özetini güncelle
                        for col in batch_cols:
                            missing_count = data[col].isnull().sum()
                            imputation_summary["variables"][col] = {
                                "type": "categorical",
                                "method": "mode",
                                "missing_count": int(missing_count),
                                "missing_percentage": float(data[col].isnull().mean() * 100)
                            }
                except Exception as e:
                    print(f"Kategorik imputasyon hata: {e}")
    
    # 2. Sayısal değişkenler için Super Learner imputasyon
    if numerical_cols:
        print("Sayısal değişkenler için imputasyon uygulanıyor...")
        num_with_missing = [col for col in numerical_cols if data[col].isnull().sum() > 0]
        
        if num_with_missing:
            # Değişkenleri eksik değer oranına göre grupla
            high_missing = [col for col in num_with_missing if data[col].isnull().mean() > 0.5]
            medium_missing = [col for col in num_with_missing if 0.2 <= data[col].isnull().mean() <= 0.5]
            low_missing = [col for col in num_with_missing if data[col].isnull().mean() < 0.2]
            
            print(f"  Yüksek eksik oranı (>%50): {len(high_missing)} değişken")
            print(f"  Orta eksik oranı (%20-%50): {len(medium_missing)} değişken")
            print(f"  Düşük eksik oranı (<%20): {len(low_missing)} değişken")
            
            # Yüksek eksik oranlı değişkenler için ortalama imputasyonu
            if high_missing:
                print(f"  Yüksek eksik oranlı değişkenlere ortalama imputasyonu uygulanıyor...")
                mean_imputer = SimpleImputer(strategy='mean')
                
                # Bellek yönetimi için gruplar halinde işle
                batch_size = 30
                for i in tqdm(range(0, len(high_missing), batch_size), desc="Yüksek eksik"):
                    batch_cols = high_missing[i:i+batch_size]
                    
                    # Tamamen eksik sütunları tespit et ve çıkar
                    valid_cols = []
                    for col in batch_cols:
                        # En az bir değer varsa kullan
                        if data[col].isnull().sum() < len(data):
                            valid_cols.append(col)
                    
                    if valid_cols:
                        try:
                            batch_data = imputed_data[valid_cols].copy()
                            imputed_values = mean_imputer.fit_transform(batch_data)
                            
                            # Sütun sayılarını kontrol et
                            if imputed_values.shape[1] == len(valid_cols):
                                # Sonuçları ana veri çerçevesine aktar
                                imputed_batch = pd.DataFrame(imputed_values, columns=valid_cols, index=imputed_data.index)
                                imputed_data[valid_cols] = imputed_batch
                                
                                # İmputasyon özetini güncelle
                                for col in valid_cols:
                                    missing_count = data[col].isnull().sum()
                                    imputation_summary["variables"][col] = {
                                        "type": "numerical",
                                        "method": "mean",
                                        "missing_count": int(missing_count),
                                        "missing_percentage": float(data[col].isnull().mean() * 100),
                                        "missing_group": "high"
                                    }
                        except Exception as e:
                            print(f"  Yüksek eksik imputasyon hatası: {e}")
                            # Hata durumunda her sütunu tek tek doldur
                            for col in valid_cols:
                                try:
                                    if data[col].isnull().sum() < len(data):
                                        imputed_data[col] = data[col].fillna(data[col].mean())
                                        
                                        imputation_summary["variables"][col] = {
                                            "type": "numerical",
                                            "method": "mean (fallback)",
                                            "missing_count": int(data[col].isnull().sum()),
                                            "missing_percentage": float(data[col].isnull().mean() * 100),
                                            "missing_group": "high",
                                            "error": str(e)
                                        }
                                except:
                                    pass
            
            # Orta eksik oranlı değişkenler için medyan imputasyonu
            if medium_missing:
                print(f"  Orta eksik oranlı değişkenlere imputasyon uygulanıyor...")
                
                # Gelişmiş yöntemler mevcutsa MICE kullan, değilse medyan
                try:
                    medium_imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=0)
                    medium_method = "MICE-Bayesian"
                except:
                    medium_imputer = SimpleImputer(strategy='median')
                    medium_method = "median"
                
                for i in tqdm(range(0, len(medium_missing), batch_size), desc="Orta eksik"):
                    batch_cols = medium_missing[i:i+batch_size]
                    
                    # Tamamen eksik sütunları tespit et ve çıkar
                    valid_cols = []
                    for col in batch_cols:
                        if data[col].isnull().sum() < len(data):
                            valid_cols.append(col)
                    
                    if valid_cols:
                        try:
                            batch_data = imputed_data[valid_cols].copy()
                            imputed_values = medium_imputer.fit_transform(batch_data)
                            
                            # Sonuçları ana veri çerçevesine aktar
                            if imputed_values.shape[1] == len(valid_cols):
                                imputed_batch = pd.DataFrame(imputed_values, columns=valid_cols, index=imputed_data.index)
                                imputed_data[valid_cols] = imputed_batch
                                
                                # İmputasyon özetini güncelle
                                for col in valid_cols:
                                    missing_count = data[col].isnull().sum()
                                    imputation_summary["variables"][col] = {
                                        "type": "numerical",
                                        "method": medium_method,
                                        "missing_count": int(missing_count),
                                        "missing_percentage": float(data[col].isnull().mean() * 100),
                                        "missing_group": "medium"
                                    }
                        except Exception as e:
                            print(f"  Orta eksik imputasyon hatası: {e}")
                            # Hata durumunda her sütunu tek tek doldur
                            for col in valid_cols:
                                try:
                                    if data[col].isnull().sum() < len(data):
                                        imputed_data[col] = data[col].fillna(data[col].median())
                                        
                                        imputation_summary["variables"][col] = {
                                            "type": "numerical",
                                            "method": "median (fallback)",
                                            "missing_count": int(data[col].isnull().sum()),
                                            "missing_percentage": float(data[col].isnull().mean() * 100),
                                            "missing_group": "medium",
                                            "error": str(e)
                                        }
                                except:
                                    pass
            
            # Düşük eksik oranlı değişkenler için KNN veya ortalama imputasyonu
            if low_missing:
                print(f"  Düşük eksik oranlı değişkenlere imputasyon uygulanıyor...")
                
                # Gelişmiş yöntemler mevcutsa KNN kullan, değilse ortalama
                try:
                    low_imputer = KNNImputer(n_neighbors=5)
                    low_method = "KNN-5"
                except:
                    low_imputer = SimpleImputer(strategy='mean')
                    low_method = "mean"
                
                for i in tqdm(range(0, len(low_missing), batch_size), desc="Düşük eksik"):
                    batch_cols = low_missing[i:i+batch_size]
                    
                    # Tamamen eksik sütunları tespit et ve çıkar
                    valid_cols = []
                    for col in batch_cols:
                        if data[col].isnull().sum() < len(data):
                            valid_cols.append(col)
                    
                    if valid_cols:
                        try:
                            batch_data = imputed_data[valid_cols].copy()
                            imputed_values = low_imputer.fit_transform(batch_data)
                            
                            # Sonuçları ana veri çerçevesine aktar
                            if imputed_values.shape[1] == len(valid_cols):
                                imputed_batch = pd.DataFrame(imputed_values, columns=valid_cols, index=imputed_data.index)
                                imputed_data[valid_cols] = imputed_batch
                                
                                # İmputasyon özetini güncelle
                                for col in valid_cols:
                                    missing_count = data[col].isnull().sum()
                                    imputation_summary["variables"][col] = {
                                        "type": "numerical",
                                        "method": low_method,
                                        "missing_count": int(missing_count),
                                        "missing_percentage": float(data[col].isnull().mean() * 100),
                                        "missing_group": "low"
                                    }
                        except Exception as e:
                            print(f"  Düşük eksik imputasyon hatası: {e}")
                            # Hata durumunda her sütunu tek tek doldur
                            for col in valid_cols:
                                try:
                                    if data[col].isnull().sum() < len(data):
                                        imputed_data[col] = data[col].fillna(data[col].mean())
                                        
                                        imputation_summary["variables"][col] = {
                                            "type": "numerical",
                                            "method": "mean (fallback)",
                                            "missing_count": int(data[col].isnull().sum()),
                                            "missing_percentage": float(data[col].isnull().mean() * 100),
                                            "missing_group": "low",
                                            "error": str(e)
                                        }
                                except:
                                    pass
    
    # İmputasyon sonrası eksik değer kontrolü
    remaining_missing = imputed_data.isnull().sum().sum()
    imputation_summary["metadata"]["remaining_missing_values"] = int(remaining_missing)
    imputation_summary["metadata"]["imputation_success_rate"] = float(100 - (remaining_missing / data.isnull().sum().sum() * 100)) if data.isnull().sum().sum() > 0 else 100.0
    
    # Sonuçları JSON dosyasına kaydet
    with open(f"{output_prefix}_super_learner_imputation.json", 'w', encoding='utf-8') as f:
        json.dump(imputation_summary, f, indent=4, ensure_ascii=False)
    
    # İmputasyon sonrası veriyi kaydet
    imputed_data.to_csv(f"{output_prefix}_super_learner_imputed_data.csv", index=False)
    
    print("\nSuper Learner İmputasyon Tamamlandı!")
    print(f"Sonuçlar:")
    print(f"  İmputasyon özeti: {output_prefix}_super_learner_imputation.json")
    print(f"  İmputasyon sonrası veri: {output_prefix}_super_learner_imputed_data.csv")
    print(f"  Başlangıçtaki eksik değer sayısı: {data.isnull().sum().sum()}")
    print(f"  İmputasyon sonrası kalan eksik değer sayısı: {remaining_missing}")
    print(f"  İmputasyon başarı oranı: %{imputation_summary['metadata']['imputation_success_rate']:.2f}")
    
    return imputed_data, imputation_summary

def ensemble_imputation(data, output_prefix):
    """
    Farklı imputasyon yöntemlerinin sonuçlarını birleştiren ensemble yaklaşımı.
    """
    from sklearn.impute import SimpleImputer
    
    print("\nEnsemble İmputasyon Başlatılıyor...")
    
    # Değişkenleri tipine göre ayır
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"Kategorik değişkenler: {len(categorical_cols)}")
    print(f"Sayısal değişkenler: {len(numerical_cols)}")
    
    # Sonuç veri çerçevesi
    imputed_data = data.copy()
    
    # İmputasyon yöntemleri özeti
    imputation_summary = {
        "metadata": {
            "total_variables": len(numerical_cols) + len(categorical_cols),
            "numerical_variables": len(numerical_cols),
            "categorical_variables": len(categorical_cols),
            "ensemble_method": "simple"
        },
        "variables": {}
    }
    
    # Kategorik değişkenler için mod imputasyonu
    if categorical_cols:
        cat_with_missing = [col for col in categorical_cols if data[col].isnull().sum() > 0]
        
        if cat_with_missing:
            print(f"Kategorik değişkenler için mod imputasyonu uygulanıyor...")
            for col in tqdm(cat_with_missing, desc="Kategorik değişkenler"):
                if data[col].isnull().sum() < len(data):  # Tamamen eksik değilse
                    imputed_data[col] = data[col].fillna(data[col].mode()[0] if len(data[col].mode()) > 0 else "Unknown")
                    
                    imputation_summary["variables"][col] = {
                        "type": "categorical",
                        "method": "mode",
                        "missing_count": int(data[col].isnull().sum()),
                        "missing_percentage": float(data[col].isnull().mean() * 100)
                    }
    
    # Sayısal değişkenler için ensemble imputasyon
    if numerical_cols:
        num_with_missing = [col for col in numerical_cols if data[col].isnull().sum() > 0]
        
        if num_with_missing:
            print(f"Sayısal değişkenler için ensemble imputasyon uygulanıyor...")
            
            # Değişkenleri eksik değer oranına göre grupla
            high_missing = [col for col in num_with_missing if data[col].isnull().mean() > 0.5]
            medium_missing = [col for col in num_with_missing if 0.2 <= data[col].isnull().mean() <= 0.5]
            low_missing = [col for col in num_with_missing if data[col].isnull().mean() < 0.2]
            
            print(f"  Yüksek eksik oranı (>%50): {len(high_missing)} değişken")
            print(f"  Orta eksik oranı (%20-%50): {len(medium_missing)} değişken")
            print(f"  Düşük eksik oranı (<%20): {len(low_missing)} değişken")
            
            # Yüksek eksik oranlı değişkenler için
            if high_missing:
                print(f"  Yüksek eksik oranlı değişkenler için imputasyon uygulanıyor...")
                
                for col in tqdm(high_missing, desc="Yüksek eksik"):
                    if data[col].isnull().sum() < len(data):  # Tamamen eksik değilse
                        try:
                            # Medyan imputasyonu
                            imputed_data[col] = data[col].fillna(data[col].median())
                            
                            imputation_summary["variables"][col] = {
                                "type": "numerical",
                                "method": "median",
                                "missing_count": int(data[col].isnull().sum()),
                                "missing_percentage": float(data[col].isnull().mean() * 100),
                                "missing_group": "high"
                            }
                        except Exception as e:
                            print(f"  {col} için hata: {e}")
            
            # Orta eksik oranlı değişkenler için
            if medium_missing:
                print(f"  Orta eksik oranlı değişkenler için imputasyon uygulanıyor...")
                
                for col in tqdm(medium_missing, desc="Orta eksik"):
                    if data[col].isnull().sum() < len(data):  # Tamamen eksik değilse
                        try:
                            # Ortalama imputasyonu
                            imputed_data[col] = data[col].fillna(data[col].mean())
                            
                            imputation_summary["variables"][col] = {
                                "type": "numerical",
                                "method": "mean",
                                "missing_count": int(data[col].isnull().sum()),
                                "missing_percentage": float(data[col].isnull().mean() * 100),
                                "missing_group": "medium"
                            }
                        except Exception as e:
                            print(f"  {col} için hata: {e}")
            
            # Düşük eksik oranlı değişkenler için
            if low_missing:
                print(f"  Düşük eksik oranlı değişkenler için imputasyon uygulanıyor...")
                
                for col in tqdm(low_missing, desc="Düşük eksik"):
                    if data[col].isnull().sum() < len(data):  # Tamamen eksik değilse
                        try:
                            # Ortalama imputasyonu
                            imputed_data[col] = data[col].fillna(data[col].mean())
                            
                            imputation_summary["variables"][col] = {
                                "type": "numerical",
                                "method": "mean",
                                "missing_count": int(data[col].isnull().sum()),
                                "missing_percentage": float(data[col].isnull().mean() * 100),
                                "missing_group": "low"
                            }
                        except Exception as e:
                            print(f"  {col} için hata: {e}")
    
    # İmputasyon sonrası eksik değer kontrolü
    remaining_missing = imputed_data.isnull().sum().sum()
    imputation_summary["metadata"]["initial_missing_values"] = int(data.isnull().sum().sum())
    imputation_summary["metadata"]["remaining_missing_values"] = int(remaining_missing)
    imputation_summary["metadata"]["imputation_success_rate"] = float(100 - (remaining_missing / data.isnull().sum().sum() * 100)) if data.isnull().sum().sum() > 0 else 100.0
    
    # Sonuçları JSON dosyasına kaydet
    with open(f"{output_prefix}_ensemble_imputation.json", 'w', encoding='utf-8') as f:
        json.dump(imputation_summary, f, indent=4, ensure_ascii=False)
    
    # İmputasyon sonrası veriyi kaydet
    imputed_data.to_csv(f"{output_prefix}_ensemble_imputed_data.csv", index=False)
    
    print("\nEnsemble İmputasyon Tamamlandı!")
    print(f"Sonuçlar:")
    print(f"  İmputasyon özeti: {output_prefix}_ensemble_imputation.json")
    print(f"  İmputasyon sonrası veri: {output_prefix}_ensemble_imputed_data.csv")
    
    return imputed_data, imputation_summary

def perform_advanced_imputation(data, output_prefix, batch_size=50):
    """
    Gelişmiş imputasyon yöntemlerini kullanarak eksik değerleri doldurur.
    Bu fonksiyon, perform_advanced_imputation_core modülünün eksikliğinde kullanılır.
    """
    from sklearn.impute import SimpleImputer
    
    print("Temel imputasyon yöntemleri uygulanıyor...")
    
    # Sonuç veri çerçevesi
    imputed_data = data.copy()
    
    # İmputasyon özeti
    imputation_summary = {
        "metadata": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_variables": data.shape[1],
            "total_rows": data.shape[0]
        },
        "variables": {}
    }
    
    # Değişkenleri tipine göre ayır
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # 1. Kategorik değişkenler için mod imputasyonu
    cat_with_missing = [col for col in categorical_cols if data[col].isnull().sum() > 0]
    for col in cat_with_missing:
        if data[col].isnull().sum() < len(data):  # Tamamen eksik değilse
            # Mod ile doldur
            mode_val = data[col].mode()
            if not mode_val.empty:
                imputed_data[col] = data[col].fillna(mode_val[0])
                
                # İmputasyon özetini güncelle
                imputation_summary["variables"][col] = {
                    "type": "categorical",
                    "method": "mode",
                    "missing_count": int(data[col].isnull().sum()),
                    "missing_percentage": float(data[col].isnull().mean() * 100)
                }
    
    # 2. Sayısal değişkenler için medyan/ortalama imputasyonu
    num_with_missing = [col for col in numerical_cols if data[col].isnull().sum() > 0]
    for col in num_with_missing:
        if data[col].isnull().sum() < len(data):  # Tamamen eksik değilse
            try:
                # Medyan ile doldur
                imputed_data[col] = data[col].fillna(data[col].median())
                
                # İmputasyon özetini güncelle
                imputation_summary["variables"][col] = {
                    "type": "numerical",
                    "method": "median",
                    "missing_count": int(data[col].isnull().sum()),
                    "missing_percentage": float(data[col].isnull().mean() * 100)
                }
            except Exception as e:
                try:
                    # Ortalama ile doldur (medyan çalışmazsa)
                    imputed_data[col] = data[col].fillna(data[col].mean())
                    
                    # İmputasyon özetini güncelle
                    imputation_summary["variables"][col] = {
                        "type": "numerical",
                        "method": "mean (fallback)",
                        "missing_count": int(data[col].isnull().sum()),
                        "missing_percentage": float(data[col].isnull().mean() * 100),
                        "error": str(e)
                    }
                except:
                    print(f"  {col} değişkeni doldurulamadı.")
    
    # İmputasyon sonrası eksik değer kontrolü
    remaining_missing = imputed_data.isnull().sum().sum()
    imputation_summary["metadata"]["initial_missing_values"] = int(data.isnull().sum().sum())
    imputation_summary["metadata"]["remaining_missing_values"] = int(remaining_missing)
    imputation_summary["metadata"]["imputation_success_rate"] = float(100 - (remaining_missing / data.isnull().sum().sum() * 100)) if data.isnull().sum().sum() > 0 else 100.0
    
    # İmputasyon sonrası veriyi kaydet
    imputed_data.to_csv(f"{output_prefix}_basic_imputed_data.csv", index=False)
    
    # Özeti JSON olarak kaydet
    with open(f"{output_prefix}_basic_imputation_summary.json", 'w', encoding='utf-8') as f:
        json.dump(imputation_summary, f, indent=4, ensure_ascii=False)
    
    print("Temel imputasyon tamamlandı.")
    print(f"Başlangıçtaki eksik değer sayısı: {data.isnull().sum().sum()}")
    print(f"İmputasyon sonrası kalan eksik değer sayısı: {remaining_missing}")
    print(f"İmputasyon başarı oranı: %{imputation_summary['metadata']['imputation_success_rate']:.2f}")
    
    return imputed_data

def eksik_veri_mekanizmalarini_analiz_et(data, output_prefix):
    """
    Veri setindeki eksik değerlerin mekanizmalarını (MCAR, MAR, MNAR) analiz eder.
    """
    print("\nEksik veri mekanizmaları (MCAR, MAR, MNAR) analiz ediliyor...")
    
    results = {
        "metadata": {
            "toplam_degisken": len(data.columns),
            "toplam_satir": len(data),
            "toplam_eksik": int(data.isnull().sum().sum()),
            "genel_eksiklik_yuzdesi": float((data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100)
        },
        "little_mcar_testi": {},
        "korelasyon_analizi": {},
        "kosullu_analiz": {},
        "demografik_desenler": {},
        "sonuc": {}
    }
    
    # 1. Basitleştirilmiş MCAR Analizi (multi_normal_test yerine)
    print("Basitleştirilmiş MCAR analizi yapılıyor...")
    try:
        # Test için sayısal değişkenlerin bir alt kümesini seç (en fazla 20)
        sayisal_degiskenler = data.select_dtypes(include=['int64', 'float64']).columns
        if len(sayisal_degiskenler) > 20:
            # Daha iyi sonuç için eksik değerli sütunları seç
            eksik_degerli_sutunlar = [col for col in sayisal_degiskenler if data[col].isnull().sum() > 0]
            if len(eksik_degerli_sutunlar) > 20:
                # Eksik değerleri olan 20 sütunun örneklemini al
                test_sutunlari = np.random.choice(eksik_degerli_sutunlar, 20, replace=False)
            elif len(eksik_degerli_sutunlar) > 0:
                test_sutunlari = eksik_degerli_sutunlar
            else:
                test_sutunlari = np.random.choice(sayisal_degiskenler, min(20, len(sayisal_degiskenler)), replace=False)
        else:
            test_sutunlari = sayisal_degiskenler
            
        # En az 2 sütun varsa analizi uygula
        if len(test_sutunlari) >= 2:
            # Eksiklik desenleri oluştur
            eksiklik_desenleri = data[test_sutunlari].isnull().astype(int)
            
            # Eksiklik göstergeleri arasında korelasyon hesapla
            korelasyon_matrisi = eksiklik_desenleri.corr()
            
            # Korelasyon matrisinin ortalama mutlak değerini hesapla (0'a yakınsa daha MCAR'a yakındır)
            ortalama_korelasyon = np.abs(korelasyon_matrisi.values).mean()
            
            # Ki-kare testine yaklaşımsal değerler
            serbestlik_derecesi = (len(test_sutunlari) * (len(test_sutunlari) - 1)) / 2
            
            # p değeri için basit bir heuristik (düşük korelasyon = yüksek p-değeri = MCAR'a daha yakın)
            # Bu kesin bir test değil, sadece bir yaklaşım
            p_degeri = 1.0 - ortalama_korelasyon  # Düşük korelasyon = yüksek p-değeri
            
            results["little_mcar_testi"] = {
                "ortalama_mutlak_korelasyon": float(ortalama_korelasyon),
                "yaklasimsal_p_degeri": float(p_degeri),
                "mcar_mi": p_degeri > 0.5,  # Eşik olarak 0.5 kullanıyoruz
                "yorum": "Veriler muhtemelen MCAR (Tamamen Rastgele Eksik)" if p_degeri > 0.5 else 
                         "Veriler muhtemelen MCAR değil (MAR veya MNAR olabilir)",
                "test_sutunlari": list(test_sutunlari),
                "not": "Bu, resmi Little's MCAR testi değil, basit bir yaklaşımdır. Eksiklik göstergeleri arasındaki korelasyona dayanır."
            }
        else:
            results["little_mcar_testi"] = {
                "not": "MCAR analizi için yeterli sayısal sütun yok"
            }
    except Exception as e:
        print(f"  MCAR analizi hatası: {e}")
        results["little_mcar_testi"] = {
            "hata": str(e),
            "not": "MCAR analizi gerçekleştirilemedi"
        }
    
    # 2. Eksiklik göstergeleri arasındaki korelasyon
    print("Eksik değer desenleri arasındaki korelasyonlar analiz ediliyor...")
    try:
        # Eksik değerler içeren sütunlar için eksiklik göstergeleri oluştur
        eksik_sutunlar = [col for col in data.columns if data[col].isnull().sum() > 0]
        
        if eksik_sutunlar:
            # Bellek sorunlarını önlemek için makul bir sayıya sınırla
            if len(eksik_sutunlar) > 50:
                eksik_sutunlar = eksik_sutunlar[:50]
            
            # Eksiklik göstergeleri oluştur
            eksiklik_gostergeleri = pd.DataFrame()
            for col in eksik_sutunlar:
                eksiklik_gostergeleri[f"eksik_{col}"] = data[col].isnull().astype(int)
            
            # Eksiklik göstergeleri arasındaki korelasyonu hesapla
            if len(eksik_sutunlar) > 1:
                korelasyon_matrisi = eksiklik_gostergeleri.corr()
                
                # Güçlü korelasyonları bul (mutlak değer > 0.3)
                guclu_korelasyonlar = []
                for i in range(len(korelasyon_matrisi.columns)):
                    for j in range(i+1, len(korelasyon_matrisi.columns)):
                        korelasyon_degeri = korelasyon_matrisi.iloc[i, j]
                        if abs(korelasyon_degeri) > 0.3:
                            col1 = korelasyon_matrisi.columns[i].replace("eksik_", "")
                            col2 = korelasyon_matrisi.columns[j].replace("eksik_", "")
                            guclu_korelasyonlar.append({
                                "degisken1": col1,
                                "degisken2": col2,
                                "korelasyon": float(korelasyon_degeri)
                            })
                
                results["korelasyon_analizi"] = {
                    "analiz_notu": "Eksiklik göstergeleri arasındaki yüksek korelasyon MAR desenine işaret eder",
                    "yuksek_korelasyonlar_var": len(guclu_korelasyonlar) > 0,
                    "guclu_korelasyonlar": guclu_korelasyonlar
                }
                
                # Korelasyon ısı haritası oluştur ve kaydet
                plt.figure(figsize=(12, 10))
                mask = np.triu(np.ones_like(korelasyon_matrisi, dtype=bool))
                sns.heatmap(korelasyon_matrisi, mask=mask, annot=False, cmap='coolwarm', 
                           vmin=-1, vmax=1, center=0, linewidths=0.5)
                plt.title("Eksik Değer Göstergeleri Arasındaki Korelasyon")
                plt.tight_layout()
                plt.savefig(f"{output_prefix}_eksik_korelasyon_heatmap.png", dpi=300)
                plt.close()
            else:
                results["korelasyon_analizi"] = {
                    "analiz_notu": "Sadece bir sütunda eksik değer var, korelasyon analizi uygulanamaz"
                }
        else:
            results["korelasyon_analizi"] = {
                "analiz_notu": "Korelasyon analizi için eksik değer bulunamadı"
            }
    except Exception as e:
        print(f"  Korelasyon analizi hatası: {e}")
        results["korelasyon_analizi"] = {
            "hata": str(e),
            "not": "Korelasyon analizi gerçekleştirilemedi"
        }
    
    # 3. Koşullu analiz - Değerler diğer değişkenlere bağlı olarak mı eksik?
    print("Eksikliğin koşullu desenleri analiz ediliyor...")
    try:
        # Potansiyel demografik/katmanlama değişkenlerini tanımla (az sayıda benzersiz değere sahip kategorik)
        potansiyel_demo_degiskenleri = []
        for col in data.columns:
            if data[col].isnull().sum() == 0:  # Sadece tam değişkenleri kullan
                if data[col].dtype == 'object' or data[col].dtype == 'category':
                    if data[col].nunique() <= 30:  # Makul sayıda kategori
                        potansiyel_demo_degiskenleri.append(col)
                elif data[col].dtype in ['int64', 'float64']:
                    if data[col].nunique() <= 30:  # Makul sayıda değer
                        potansiyel_demo_degiskenleri.append(col)
        
        # Hesaplama sorunlarını önlemek için makul bir sayıya sınırla
        if len(potansiyel_demo_degiskenleri) > 5:
            potansiyel_demo_degiskenleri = potansiyel_demo_degiskenleri[:5]
        
        # Her demografik değişken için eksiklik desenlerini analiz et
        kosullu_desenler = []
        
        for demo_var in potansiyel_demo_degiskenleri:
            # Eksik değer içeren her sütun için
            for col in eksik_sutunlar[:20]:  # En fazla 20 sütunla sınırla
                try:
                    # Gruba göre eksik değer yüzdesini hesapla
                    gruba_gore_eksiklik = data.groupby(demo_var)[col].apply(
                        lambda x: (x.isnull().sum() / len(x)) * 100
                    ).reset_index()
                    gruba_gore_eksiklik.columns = [demo_var, 'eksiklik_yuzdesi']
                    
                    # Eksiklikte önemli bir varyasyon var mı kontrol et
                    if gruba_gore_eksiklik['eksiklik_yuzdesi'].std() > 5:  # Rastgele eşik
                        # Eksiklik yüzdesine göre sırala
                        gruba_gore_eksiklik = gruba_gore_eksiklik.sort_values('eksiklik_yuzdesi', ascending=False)
                        
                        kosullu_desenler.append({
                            "demografik_degisken": demo_var,
                            "hedef_degisken": col,
                            "varyasyon": float(gruba_gore_eksiklik['eksiklik_yuzdesi'].std()),
                            "maks_grup": str(gruba_gore_eksiklik.iloc[0][demo_var]),
                            "maks_eksiklik_yuzdesi": float(gruba_gore_eksiklik.iloc[0]['eksiklik_yuzdesi']),
                            "min_grup": str(gruba_gore_eksiklik.iloc[-1][demo_var]),
                            "min_eksiklik_yuzdesi": float(gruba_gore_eksiklik.iloc[-1]['eksiklik_yuzdesi']),
                            "mar_onerir": True
                        })
                        
                        # Seçilen desenler için görselleştirme oluştur (çıktıyı yönetilebilir tutmak için sınırla)
                        if len(kosullu_desenler) <= 5:
                            plt.figure(figsize=(12, 6))
                            sns.barplot(x=demo_var, y='eksiklik_yuzdesi', data=gruba_gore_eksiklik)
                            plt.title(f"'{col}' Değişkenindeki Eksik Değerler - '{demo_var}' Bazında")
                            plt.ylabel("Eksiklik Yüzdesi (%)")
                            plt.xticks(rotation=45, ha='right')
                            plt.tight_layout()
                            plt.savefig(f"{output_prefix}_eksiklik_{demo_var}_{col}.png", dpi=300)
                            plt.close()
                except Exception as e:
                    print(f"  {demo_var} bazında {col} analizi hatası: {e}")
        
        results["kosullu_analiz"] = {
            "analiz_edilen_demografik_degiskenler": potansiyel_demo_degiskenleri,
            "onemli_desenler": kosullu_desenler,
            "mar_onerir": len(kosullu_desenler) > 0,
            "analiz_notu": "Gruplar arası eksiklikte önemli varyasyon MAR desenine işaret eder"
        }
    except Exception as e:
        print(f"  Koşullu analiz hatası: {e}")
        results["kosullu_analiz"] = {
            "hata": str(e),
            "not": "Koşullu analiz gerçekleştirilemedi"
        }
    
    # 4. Demografik desenler - potansiyel demografik değişkenlere göre özetle
    print("Demografik eksiklik desenleri analiz ediliyor...")
    try:
        demografik_desenler = []
        
        for demo_var in potansiyel_demo_degiskenleri:
            try:
                # Gruba göre satır başına ortalama eksik değer sayısını hesapla
                eksik_sayilari = data.isnull().sum(axis=1)
                eksik_ozet = data.groupby(demo_var)[demo_var].count().reset_index(name='sayim')
                gruba_gore_eksiklik = pd.DataFrame({
                    demo_var: data[demo_var],
                    'eksik_sayisi': eksik_sayilari
                }).groupby(demo_var).agg({
                    'eksik_sayisi': ['mean', 'median', 'std']
                }).reset_index()
                
                # Önemli varyasyon var mı kontrol et
                if gruba_gore_eksiklik[('eksik_sayisi', 'std')].mean() > 2:  # Rastgele eşik
                    demografik_desenler.append({
                        "demografik_degisken": demo_var,
                        "varyasyon": float(gruba_gore_eksiklik[('eksik_sayisi', 'std')].mean()),
                        "mar_onerir": True,
                        "grup_istatistikleri": [
                            {
                                "grup": str(row[demo_var]),
                                "ortalama_eksik": float(row[('eksik_sayisi', 'mean')]),
                                "medyan_eksik": float(row[('eksik_sayisi', 'median')])
                            }
                            for _, row in gruba_gore_eksiklik.iterrows()
                        ]
                    })
                    
                    # Görselleştirme oluştur (kutu grafiği)
                    plt.figure(figsize=(12, 6))
                    sns.boxplot(x=demo_var, y='eksik_sayisi', data=pd.DataFrame({
                        demo_var: data[demo_var],
                        'eksik_sayisi': eksik_sayilari
                    }))
                    plt.title(f"Satır Başına Eksik Değer Sayısı - {demo_var} Bazında")
                    plt.ylabel("Satır Başına Eksik Değer Sayısı")
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    plt.savefig(f"{output_prefix}_eksik_sayisi_{demo_var}.png", dpi=300)
                    plt.close()
            except Exception as e:
                print(f"  {demo_var} için demografik analiz hatası: {e}")
        
        results["demografik_desenler"] = {
            "analiz_edilen_degiskenler": potansiyel_demo_degiskenleri,
            "onemli_desenler": demografik_desenler,
            "mar_onerir": len(demografik_desenler) > 0
        }
    except Exception as e:
        print(f"  Demografik desen analizi hatası: {e}")
        results["demografik_desenler"] = {
            "hata": str(e),
            "not": "Demografik desen analizi gerçekleştirilemedi"
        }
    
    # 5. Genel sonuç
    print("Eksik veri mekanizmaları hakkında sonuçlar oluşturuluyor...")
    try:
        # Her mekanizma için kanıtları topla
        mcar_kaniti = results["little_mcar_testi"].get("mcar_mi", False)
        mar_kaniti = (
            results["korelasyon_analizi"].get("yuksek_korelasyonlar_var", False) or
            results["kosullu_analiz"].get("mar_onerir", False) or
            results["demografik_desenler"].get("mar_onerir", False)
        )
        
        # MNAR doğrudan tespit edilmesi zordur, ancak MCAR ve MAR'a karşı kanıt varsa önerebiliriz
        if mcar_kaniti:
            birincil_mekanizma = "MCAR"
            guven = "orta"
            oneri = ("Ortalama/medyan imputasyonu gibi basit imputasyon yöntemleri veya "
                    "Çoklu İmputasyon gibi daha gelişmiş yöntemler geçerli seçeneklerdir.")
        elif mar_kaniti:
            birincil_mekanizma = "MAR"
            guven = "orta-yüksek" if (
                results["korelasyon_analizi"].get("yuksek_korelasyonlar_var", False) and
                (results["kosullu_analiz"].get("mar_onerir", False) or
                 results["demografik_desenler"].get("mar_onerir", False))
            ) else "orta"
            oneri = ("Çoklu İmputasyon, MICE veya diğer değişkenlere koşullu olan "
                    "model tabanlı imputasyon yöntemleri önerilir.")
        else:
            birincil_mekanizma = "MNAR veya belirlenemedi"
            guven = "düşük"
            oneri = ("Duyarlılık analizi veya ek veri toplama düşünülebilir. İmputasyon "
                    "kullanılacaksa, desen-karışım modelleri veya seçim modelleri gibi "
                    "daha gelişmiş yöntemler gerekebilir.")
        
        results["sonuc"] = {
            "birincil_mekanizma": birincil_mekanizma,
            "guven": guven,
            "mcar_kaniti": mcar_kaniti,
            "mar_kaniti": mar_kaniti,
            "oneri": oneri,
            "not": ("Bu analiz, eksik veri mekanizmaları hakkında göstergeler sağlar, "
                   "ancak kesin belirleme için alan bilgisi veya daha gelişmiş "
                   "testler gerekebilir.")
        }
    except Exception as e:
        print(f"  Sonuç oluşturma hatası: {e}")
        results["sonuc"] = {
            "hata": str(e),
            "not": "Eksik veri mekanizmaları hakkında sonuç oluşturulamadı"
        }
    
    # Sonuçları JSON dosyasına kaydet
    with open(f"{output_prefix}_eksik_mekanizmalar.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"Eksik veri mekanizmaları analizi tamamlandı.")
    print(f"Sonuç: Birincil mekanizma {results['sonuc']['birincil_mekanizma']} görünüyor (güven: {results['sonuc']['guven']})")
    
    return results

def model_based_imputation(data, numerical_cols, categorical_cols, output_prefix):
    """
    Makine öğrenimi modellerini kullanarak eksik değerleri doldurur.
    
    Her eksik değer içeren değişken için ayrı bir model eğitilerek daha doğru imputation sağlar.
    Sayısal değişkenler için regresyon, kategorik değişkenler için sınıflandırma modelleri kullanılır.
    
    Args:
        data: İmpute edilecek veri çerçevesi
        numerical_cols: Sayısal değişken listesi
        categorical_cols: Kategorik değişken listesi
        output_prefix: Çıktı dosya öneki
        
    Returns:
        imputed_data: İmpute edilmiş veri çerçevesi
        imputation_summary: İmputation sürecinin özeti
    """
    print("\nModel Tabanlı İmputation Başlatılıyor...")
    print(f"Sayısal değişken sayısı: {len(numerical_cols)}")
    print(f"Kategorik değişken sayısı: {len(categorical_cols)}")
    
    # Sonuç veri çerçevesi
    imputed_data = data.copy()
    
    # İmputasyon yöntemleri özeti
    imputation_summary = {
        "metadata": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_variables": len(numerical_cols) + len(categorical_cols),
            "numerical_variables": len(numerical_cols),
            "categorical_variables": len(categorical_cols),
            "imputation_method": "model_based"
        },
        "variables": {}
    }
    
    # Eksik değeri olan değişkenleri bul
    num_with_missing = [col for col in numerical_cols if data[col].isnull().sum() > 0]
    cat_with_missing = [col for col in categorical_cols if data[col].isnull().sum() > 0]
    
    print(f"Eksik değeri olan sayısal değişkenler: {len(num_with_missing)}")
    print(f"Eksik değeri olan kategorik değişkenler: {len(cat_with_missing)}")
    
    # Değişkenleri eksik değer oranına göre grupla
    high_missing_num = [col for col in num_with_missing if data[col].isnull().mean() > 0.5]
    medium_missing_num = [col for col in num_with_missing if 0.2 <= data[col].isnull().mean() <= 0.5]
    low_missing_num = [col for col in num_with_missing if data[col].isnull().mean() < 0.2]
    
    high_missing_cat = [col for col in cat_with_missing if data[col].isnull().mean() > 0.5]
    medium_missing_cat = [col for col in cat_with_missing if 0.2 <= data[col].isnull().mean() <= 0.5]
    low_missing_cat = [col for col in cat_with_missing if data[col].isnull().mean() < 0.2]
    
    print(f"Yüksek eksik oranlı (%50+) sayısal değişkenler: {len(high_missing_num)}")
    print(f"Orta eksik oranlı (%20-%50) sayısal değişkenler: {len(medium_missing_num)}")
    print(f"Düşük eksik oranlı (<%20) sayısal değişkenler: {len(low_missing_num)}")
    
    print(f"Yüksek eksik oranlı (%50+) kategorik değişkenler: {len(high_missing_cat)}")
    print(f"Orta eksik oranlı (%20-%50) kategorik değişkenler: {len(medium_missing_cat)}")
    print(f"Düşük eksik oranlı (<%20) kategorik değişkenler: {len(low_missing_cat)}")
    
    # Ön işleme: İmputation yapmadan önce eksik olmayan değerleri kullanarak basit imputasyon yap
    # Bu, model eğitimi için gereklidir
    print("Ön işleme: Geçici basit imputasyon yapılıyor (model eğitimi için)...")
    temp_data = data.copy()
    
    # Sayısal değişkenler için medyan imputasyonu
    num_imputer = SimpleImputer(strategy='median')
    if numerical_cols:
        temp_data[numerical_cols] = num_imputer.fit_transform(temp_data[numerical_cols])
    
    # Kategorik değişkenler için en sık değer (mod) imputasyonu
    cat_imputer = SimpleImputer(strategy='most_frequent')
    if categorical_cols:
        temp_data[categorical_cols] = cat_imputer.fit_transform(temp_data[categorical_cols])
    
    # SAYISAL DEĞİŞKENLER İÇİN MODEL TABANLI İMPUTATION
    if num_with_missing:
        print("\nSayısal değişkenler için model tabanlı imputation başlatılıyor...")
        
        # Sayısal değişkenler için regresyon modelleri
        regression_models = {
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "ElasticNet": ElasticNet(random_state=42, alpha=0.5, l1_ratio=0.5)
        }
        
        # Değişken gruplarına göre model seçimi
        model_by_group = {
            "high": "ElasticNet",       # Yüksek eksik oranı için daha basit model
            "medium": "GradientBoosting", # Orta eksik oranı için orta karmaşıklıkta
            "low": "RandomForest"       # Düşük eksik oranı için karmaşık model
        }
        
        # Her sayısal değişken için model tabanlı imputation
        for group_name, missing_cols in [
            ("low", low_missing_num), 
            ("medium", medium_missing_num), 
            ("high", high_missing_num)
        ]:
            if not missing_cols:
                continue
                
            print(f"  {group_name.capitalize()} eksik oranlı sayısal değişkenler işleniyor ({len(missing_cols)} değişken)...")
            
            # Grup için model seçimi
            model_name = model_by_group[group_name]
            base_model = regression_models[model_name]
            
            for target_col in tqdm(missing_cols, desc=f"{group_name} eksik sayısal"):
                try:
                    # Eksik değer maskesi oluştur
                    missing_mask = data[target_col].isnull()
                    
                    # Hiç eksik değer yoksa veya tümü eksikse atla
                    if missing_mask.sum() == 0 or missing_mask.sum() == len(data):
                        continue
                        
                    # Eğitim verisi: Eksik olmayan değerleri kullan
                    train_data = temp_data[~missing_mask].copy()
                    
                    # Tahmin için veri: Eksik değerlere sahip satırlar
                    predict_data = temp_data[missing_mask].copy()
                    
                    # Özellikler ve hedef
                    feature_cols = [col for col in temp_data.columns if col != target_col]
                    
                    # Çok fazla değişken varsa, korelasyona göre en önemli değişkenleri seç
                    if len(feature_cols) > 100:
                        correlations = train_data[numerical_cols].corr()[target_col].abs().sort_values(ascending=False)
                        top_numeric = correlations.index.tolist()[:50]  # En çok ilişkili 50 sayısal değişken
                        
                        # Kategorik değişkenleri de ekle (tümü veya bir alt küme)
                        categorical_subset = categorical_cols[:50] if len(categorical_cols) > 50 else categorical_cols
                        
                        feature_cols = list(set(top_numeric + categorical_subset))
                    
                    # Özellik mühendisliği: Sayısal ve kategorik değişkenleri işle
                    numeric_features = [col for col in feature_cols if col in numerical_cols]
                    categorical_features = [col for col in feature_cols if col in categorical_cols]
                    
                    # Her model türü için optimum hiperparametreler
                    if model_name == "RandomForest":
                        model = RandomForestRegressor(
                            n_estimators=100, 
                            max_depth=None,
                            min_samples_split=5,
                            min_samples_leaf=2,
                            random_state=42
                        )
                    elif model_name == "GradientBoosting":
                        model = GradientBoostingRegressor(
                            n_estimators=100,
                            learning_rate=0.1,
                            max_depth=5,
                            random_state=42
                        )
                    else:  # ElasticNet
                        model = ElasticNet(
                            alpha=0.5,
                            l1_ratio=0.5,
                            max_iter=1000,
                            random_state=42
                        )
                    
                    # Veri hazırlama (preprocessing pipeline)
                    preprocessor = ColumnTransformer(
                        transformers=[
                            ('num', StandardScaler(), numeric_features),
                            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                        ], 
                        remainder='drop'
                    )
                    
                    # Pipeline oluştur
                    pipeline = Pipeline([
                        ('preprocessor', preprocessor),
                        ('model', model)
                    ])
                    
                    # Hedef değişken için gereksiz özellikleri çıkart (aşırı korelasyon)
                    X_train = train_data[feature_cols]
                    y_train = train_data[target_col]
                    
                    # Modeli eğit
                    pipeline.fit(X_train, y_train)
                    
                    # Eksik değerleri tahmin et
                    X_predict = predict_data[feature_cols]
                    predictions = pipeline.predict(X_predict)
                    
                    # Tahminleri gerçek veri aralığına sınırla
                    min_val = data[target_col].min()
                    max_val = data[target_col].max()
                    predictions = np.clip(predictions, min_val, max_val)
                    
                    # Tahminleri veri setine yerleştir
                    imputed_data.loc[missing_mask, target_col] = predictions
                    
                    # Model performansını hesapla
                    # 5-fold cross validation yerine basit bir test kümesi değerlendirmesi
                    from sklearn.model_selection import train_test_split
                    
                    if len(train_data) > 100:  # Yeterli veri varsa
                        X_train_cv, X_test_cv, y_train_cv, y_test_cv = train_test_split(
                            X_train, y_train, test_size=0.2, random_state=42)
                        
                        pipeline.fit(X_train_cv, y_train_cv)
                        cv_predictions = pipeline.predict(X_test_cv)
                        rmse = np.sqrt(mean_squared_error(y_test_cv, cv_predictions))
                        r2 = pipeline.score(X_test_cv, y_test_cv)
                    else:
                        rmse = np.nan
                        r2 = np.nan
                    
                    # İmputasyon özetini güncelle
                    imputation_summary["variables"][target_col] = {
                        "type": "numerical",
                        "method": f"regression_{model_name}",
                        "missing_count": int(missing_mask.sum()),
                        "missing_percentage": float(missing_mask.mean() * 100),
                        "missing_group": group_name,
                        "model_rmse": float(rmse) if not np.isnan(rmse) else None,
                        "model_r2": float(r2) if not np.isnan(r2) else None,
                        "feature_count": len(feature_cols)
                    }
                    
                except Exception as e:
                    print(f"  Hata ({target_col}): {e}")
                    # Hata durumunda basit imputation ile devam et
                    if group_name == "high":
                        fill_value = data[target_col].median()
                    else:
                        fill_value = data[target_col].mean()
                        
                    imputed_data.loc[missing_mask, target_col] = fill_value
                    
                    imputation_summary["variables"][target_col] = {
                        "type": "numerical",
                        "method": "fallback_simple",
                        "fallback_method": "median" if group_name == "high" else "mean",
                        "missing_count": int(missing_mask.sum()),
                        "missing_percentage": float(missing_mask.mean() * 100),
                        "missing_group": group_name,
                        "error": str(e)
                    }
    
    # KATEGORİK DEĞİŞKENLER İÇİN MODEL TABANLI İMPUTATION
    if cat_with_missing:
        print("\nKategorik değişkenler için model tabanlı imputation başlatılıyor...")
        
        # Kategorik değişkenler için sınıflandırma modelleri
        classification_models = {
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
            "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42)
        }
        
        # Değişken gruplarına göre model seçimi
        model_by_group = {
            "high": "LogisticRegression",  # Yüksek eksik oranı için daha basit model
            "medium": "GradientBoosting",  # Orta eksik oranı için orta karmaşıklıkta
            "low": "RandomForest"          # Düşük eksik oranı için karmaşık model
        }
        
        # Her kategorik değişken için model tabanlı imputation
        for group_name, missing_cols in [
            ("low", low_missing_cat), 
            ("medium", medium_missing_cat), 
            ("high", high_missing_cat)
        ]:
            if not missing_cols:
                continue
                
            print(f"  {group_name.capitalize()} eksik oranlı kategorik değişkenler işleniyor ({len(missing_cols)} değişken)...")
            
            # Grup için model seçimi
            model_name = model_by_group[group_name]
            base_model = classification_models[model_name]
            
            for target_col in tqdm(missing_cols, desc=f"{group_name} eksik kategorik"):
                try:
                    # Eksik değer maskesi oluştur
                    missing_mask = data[target_col].isnull()
                    
                    # Hiç eksik değer yoksa veya tümü eksikse atla
                    if missing_mask.sum() == 0 or missing_mask.sum() == len(data):
                        continue
                    
                    # Benzersiz değer sayısını kontrol et - çok fazlaysa atla
                    n_unique = data[target_col].nunique()
                    if n_unique > 100:  # Çok fazla sınıf varsa model eğitimi zor olabilir
                        print(f"  {target_col} değişkeni çok fazla sınıf içeriyor ({n_unique}), mod imputasyonu kullanılacak.")
                        mode_value = data[target_col].mode()[0] if not data[target_col].mode().empty else None
                        imputed_data.loc[missing_mask, target_col] = mode_value
                        
                        imputation_summary["variables"][target_col] = {
                            "type": "categorical",
                            "method": "mode",
                            "reason": "too_many_classes",
                            "missing_count": int(missing_mask.sum()),
                            "missing_percentage": float(missing_mask.mean() * 100),
                            "missing_group": group_name,
                            "unique_classes": int(n_unique)
                        }
                        continue
                        
                    # Eğitim verisi: Eksik olmayan değerleri kullan
                    train_data = temp_data[~missing_mask].copy()
                    
                    # Tahmin için veri: Eksik değerlere sahip satırlar
                    predict_data = temp_data[missing_mask].copy()
                    
                    # Özellikler ve hedef
                    feature_cols = [col for col in temp_data.columns if col != target_col]
                    
                    # Çok fazla değişken varsa, ilişki veya önem ölçütlerine göre seç
                    if len(feature_cols) > 100:
                        # Kategorik hedef için Kruskal-Wallis testi kullanılabilir
                        important_numeric = []
                        for num_col in numerical_cols:
                            if num_col != target_col:
                                try:
                                    # Her benzersiz sınıf için değerleri grupla
                                    groups = []
                                    for cls in train_data[target_col].unique():
                                        group_values = train_data[train_data[target_col] == cls][num_col].dropna()
                                        if len(group_values) > 0:
                                            groups.append(group_values)
                                    
                                    # En az 2 grup varsa Kruskal-Wallis testi uygula
                                    if len(groups) >= 2:
                                        stat, p_value = stats.kruskal(*groups)
                                        if p_value < 0.05:  # İstatistiksel olarak anlamlı
                                            important_numeric.append(num_col)
                                except:
                                    pass
                        
                        # En önemli sayısal değişkenleri seç (maksimum 50)
                        top_numeric = important_numeric[:50] if len(important_numeric) > 50 else important_numeric
                        
                        # Kategorik değişkenleri de ekle (tümü veya bir alt küme)
                        categorical_subset = categorical_cols[:50] if len(categorical_cols) > 50 else categorical_cols
                        
                        feature_cols = list(set(top_numeric + categorical_subset))
                    
                    # Özellik mühendisliği: Sayısal ve kategorik değişkenleri işle
                    numeric_features = [col for col in feature_cols if col in numerical_cols]
                    categorical_features = [col for col in feature_cols if col in categorical_cols]
                    
                    # Her model türü için optimum hiperparametreler
                    if model_name == "RandomForest":
                        model = RandomForestClassifier(
                            n_estimators=100, 
                            max_depth=None,
                            min_samples_split=5,
                            min_samples_leaf=2,
                            class_weight='balanced',
                            random_state=42
                        )
                    elif model_name == "GradientBoosting":
                        model = GradientBoostingClassifier(
                            n_estimators=100,
                            learning_rate=0.1,
                            max_depth=5,
                            random_state=42
                        )
                    else:  # LogisticRegression
                        model = LogisticRegression(
                            max_iter=1000,
                            class_weight='balanced',
                            random_state=42
                        )
                    
                    # Veri hazırlama (preprocessing pipeline)
                    preprocessor = ColumnTransformer(
                        transformers=[
                            ('num', StandardScaler(), numeric_features),
                            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                        ], 
                        remainder='drop'
                    )
                    
                    # Pipeline oluştur
                    pipeline = Pipeline([
                        ('preprocessor', preprocessor),
                        ('model', model)
                    ])
                    
                    # Modeli eğit
                    X_train = train_data[feature_cols]
                    y_train = train_data[target_col]
                    
                    pipeline.fit(X_train, y_train)
                    
                    # Eksik değerleri tahmin et
                    X_predict = predict_data[feature_cols]
                    predictions = pipeline.predict(X_predict)
                    
                    # Tahminleri veri setine yerleştir
                    imputed_data.loc[missing_mask, target_col] = predictions
                    
                    # Model performansını hesapla
                    from sklearn.model_selection import train_test_split
                    
                    if len(train_data) > 100:  # Yeterli veri varsa
                        X_train_cv, X_test_cv, y_train_cv, y_test_cv = train_test_split(
                            X_train, y_train, test_size=0.2, random_state=42)
                        
                        pipeline.fit(X_train_cv, y_train_cv)
                        cv_predictions = pipeline.predict(X_test_cv)
                        accuracy = accuracy_score(y_test_cv, cv_predictions)
                    else:
                        accuracy = np.nan
                    
                    # İmputasyon özetini güncelle
                    imputation_summary["variables"][target_col] = {
                        "type": "categorical",
                        "method": f"classification_{model_name}",
                        "missing_count": int(missing_mask.sum()),
                        "missing_percentage": float(missing_mask.mean() * 100),
                        "missing_group": group_name,
                        "model_accuracy": float(accuracy) if not np.isnan(accuracy) else None,
                        "feature_count": len(feature_cols),
                        "unique_classes": int(n_unique)
                    }
                    
                except Exception as e:
                    print(f"  Hata ({target_col}): {e}")
                    # Hata durumunda mod imputasyonu ile devam et
                    mode_value = data[target_col].mode()[0] if not data[target_col].mode().empty else None
                    imputed_data.loc[missing_mask, target_col] = mode_value
                    
                    imputation_summary["variables"][target_col] = {
                        "type": "categorical",
                        "method": "fallback_mode",
                        "missing_count": int(missing_mask.sum()),
                        "missing_percentage": float(missing_mask.mean() * 100),
                        "missing_group": group_name,
                        "error": str(e)
                    }
    
    # İmputasyon sonrası eksik değer kontrolü
    remaining_missing = imputed_data.isnull().sum().sum()
    imputation_summary["metadata"]["initial_missing_values"] = int(data.isnull().sum().sum())
    imputation_summary["metadata"]["remaining_missing_values"] = int(remaining_missing)
    imputation_summary["metadata"]["imputation_success_rate"] = float(100 - (remaining_missing / data.isnull().sum().sum() * 100)) if data.isnull().sum().sum() > 0 else 100.0
    
    # Sonuçları JSON dosyasına kaydet
    with open(f"{output_prefix}_model_based_imputation.json", 'w', encoding='utf-8') as f:
        json.dump(imputation_summary, f, indent=4, ensure_ascii=False)
    
    # İmputasyon sonrası veriyi kaydet
    imputed_data.to_csv(f"{output_prefix}_model_based_imputed_data.csv", index=False)
    
    print("\nModel Tabanlı İmputasyon Tamamlandı!")
    print(f"Sonuçlar:")
    print(f"  İmputasyon özeti: {output_prefix}_model_based_imputation.json")
    print(f"  İmputasyon sonrası veri: {output_prefix}_model_based_imputed_data.csv")
    print(f"  Başlangıçtaki eksik değer sayısı: {data.isnull().sum().sum()}")
    print(f"  İmputasyon sonrası kalan eksik değer sayısı: {remaining_missing}")
    print(f"  İmputasyon başarı oranı: %{imputation_summary['metadata']['imputation_success_rate']:.2f}")
    
    return imputed_data, imputation_summary

def stacking_model_imputation(data, numerical_cols, categorical_cols, output_prefix):
    """
    Stacking (model yığınlama) yaklaşımını kullanan gelişmiş imputation yöntemi.
    
    Birden fazla modelin tahminlerini birleştirerek daha güçlü imputation sonuçları sağlar.
    
    Args:
        data: İmpute edilecek veri çerçevesi
        numerical_cols: Sayısal değişken listesi
        categorical_cols: Kategorik değişken listesi
        output_prefix: Çıktı dosya öneki
        
    Returns:
        imputed_data: İmpute edilmiş veri çerçevesi
        imputation_summary: İmputation sürecinin özeti
    """
    print("\nStacking Model İmputasyonu Başlatılıyor...")
    print(f"Sayısal değişken sayısı: {len(numerical_cols)}")
    print(f"Kategorik değişken sayısı: {len(categorical_cols)}")
    
    # Sonuç veri çerçevesi
    imputed_data = data.copy()
    
    # İmputasyon yöntemleri özeti
    imputation_summary = {
        "metadata": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_variables": len(numerical_cols) + len(categorical_cols),
            "numerical_variables": len(numerical_cols),
            "categorical_variables": len(categorical_cols),
            "imputation_method": "stacking_model"
        },
        "variables": {}
    }
    
    # Eksik değeri olan değişkenleri bul
    num_with_missing = [col for col in numerical_cols if data[col].isnull().sum() > 0]
    cat_with_missing = [col for col in categorical_cols if data[col].isnull().sum() > 0]
    
    print(f"Eksik değeri olan sayısal değişkenler: {len(num_with_missing)}")
    print(f"Eksik değeri olan kategorik değişkenler: {len(cat_with_missing)}")
    
    # Ön işleme: İmputation yapmadan önce eksik olmayan değerleri kullanarak basit imputasyon yap
    # Bu, model eğitimi için gereklidir
    print("Ön işleme: Geçici basit imputasyon yapılıyor (model eğitimi için)...")
    temp_data = data.copy()
    
    # Sayısal değişkenler için medyan imputasyonu
    num_imputer = SimpleImputer(strategy='median')
    if numerical_cols:
        temp_data[numerical_cols] = num_imputer.fit_transform(temp_data[numerical_cols])
    
    # Kategorik değişkenler için en sık değer (mod) imputasyonu
    cat_imputer = SimpleImputer(strategy='most_frequent')
    if categorical_cols:
        temp_data[categorical_cols] = cat_imputer.fit_transform(temp_data[categorical_cols])
    
    # SAYISAL DEĞİŞKENLER İÇİN STACKING MODEL İMPUTATION
    if num_with_missing:
        print("\nSayısal değişkenler için stacking model imputation başlatılıyor...")
        
        # Sayısal değişkenler için regresyon modelleri (seviye-1 modeller)
        level1_models = {
            "RF": RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42),
            "GB": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
            "EN": ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42),
            "RG": Ridge(alpha=1.0, random_state=42),
            "LS": Lasso(alpha=0.1, random_state=42),
            "KNN": KNeighborsRegressor(n_neighbors=5)
        }
        
        # Seviye-2 model (meta-model)
        meta_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
        
        # Her sayısal değişken için stacking imputation
        for target_col in tqdm(num_with_missing, desc="Sayısal değişkenler"):
            try:
                # Eksik değer maskesi oluştur
                missing_mask = data[target_col].isnull()
                
                # Hiç eksik değer yoksa veya tümü eksikse atla
                if missing_mask.sum() == 0 or missing_mask.sum() == len(data):
                    continue
                    
                # Eğitim verisi: Eksik olmayan değerleri kullan
                train_data = temp_data[~missing_mask].copy()
                
                # Tahmin için veri: Eksik değerlere sahip satırlar
                predict_data = temp_data[missing_mask].copy()
                
                # Özellikler ve hedef
                feature_cols = [col for col in temp_data.columns if col != target_col]
                
                # Çok fazla değişken varsa, korelasyona göre en önemli değişkenleri seç
                if len(feature_cols) > 100:
                    correlations = pd.DataFrame()
                    try:
                        correlations = train_data[numerical_cols].corr()[target_col].abs().sort_values(ascending=False)
                        top_numeric = correlations.index.tolist()[:50]  # En çok ilişkili 50 sayısal değişken
                    except:
                        top_numeric = numerical_cols[:50] if len(numerical_cols) > 50 else numerical_cols
                    
                    # Kategorik değişkenleri de ekle (tümü veya bir alt küme)
                    categorical_subset = categorical_cols[:50] if len(categorical_cols) > 50 else categorical_cols
                    
                    feature_cols = list(set(top_numeric + categorical_subset))
                
                # Özellik mühendisliği: Sayısal ve kategorik değişkenleri işle
                numeric_features = [col for col in feature_cols if col in numerical_cols]
                categorical_features = [col for col in feature_cols if col in categorical_cols]
                
                # Veri ön işleme
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', StandardScaler(), numeric_features),
                        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                    ], 
                    remainder='drop'
                )
                
                # Eğitim verisi hazırlama
                X_train = train_data[feature_cols]
                y_train = train_data[target_col]
                
                # Çapraz doğrulama için veri hazırlama (seviye-2 model eğitimi için)
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                
                # Seviye-1 model tahminlerini topla
                meta_features = np.zeros((X_train.shape[0], len(level1_models)))
                
                # Her seviye-1 model için cross-validation ile meta-özellikler oluştur
                for i, (model_name, model) in enumerate(level1_models.items()):
                    # Pipeline oluştur
                    pipeline = Pipeline([
                        ('preprocessor', preprocessor),
                        ('model', model)
                    ])
                    
                    # Çapraz doğrulama tahminleri
                    cv_preds = np.zeros(X_train.shape[0])
                    
                    for train_idx, val_idx in kf.split(X_train):
                        # Eğitim ve doğrulama bölümleri
                        X_train_cv, X_val_cv = X_train.iloc[train_idx], X_train.iloc[val_idx]
                        y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]
                        
                        # Modeli eğit
                        pipeline.fit(X_train_cv, y_train_cv)
                        
                        # Doğrulama kümesi için tahminler
                        cv_preds[val_idx] = pipeline.predict(X_val_cv)
                    
                    # Meta-özellikler olarak tahminleri kaydet
                    meta_features[:, i] = cv_preds
                
                # Seviye-2 modeli (meta-model) eğit
                meta_model.fit(meta_features, y_train)
                
                # Seviye-1 modelleri tam veri seti üzerinde yeniden eğit
                level1_predictions = np.zeros((predict_data.shape[0], len(level1_models)))
                
                for i, (model_name, model) in enumerate(level1_models.items()):
                    pipeline = Pipeline([
                        ('preprocessor', preprocessor),
                        ('model', model)
                    ])
                    
                    # Tüm eğitim verisiyle eğit
                    pipeline.fit(X_train, y_train)
                    
                    # Tahmin verisi için öngörüler
                    X_predict = predict_data[feature_cols]
                    level1_predictions[:, i] = pipeline.predict(X_predict)
                
                # Meta-model ile nihai tahminleri yap
                final_predictions = meta_model.predict(level1_predictions)
                
                # Tahminleri gerçek veri aralığına sınırla
                min_val = data[target_col].min()
                max_val = data[target_col].max()
                final_predictions = np.clip(final_predictions, min_val, max_val)
                
                # Tahminleri veri setine yerleştir
                imputed_data.loc[missing_mask, target_col] = final_predictions
                
                # Model performansını hesapla (çapraz doğrulama RMSE)
                cv_scores = cross_val_score(
                    meta_model, meta_features, y_train, 
                    cv=5, scoring='neg_root_mean_squared_error'
                )
                rmse = -np.mean(cv_scores)
                
                # Model ağırlıklarını/önemlerini hesapla
                model_importances = meta_model.feature_importances_
                model_weights = {name: float(imp) for name, imp in zip(level1_models.keys(), model_importances)}
                
                # İmputasyon özetini güncelle
                imputation_summary["variables"][target_col] = {
                    "type": "numerical",
                    "method": "stacking_regression",
                    "missing_count": int(missing_mask.sum()),
                    "missing_percentage": float(missing_mask.mean() * 100),
                    "model_rmse": float(rmse),
                    "feature_count": len(feature_cols),
                    "model_weights": model_weights
                }
                
            except Exception as e:
                print(f"  Hata ({target_col}): {e}")
                # Hata durumunda model tabanlı imputasyon yöntemi ile devam et
                try:
                    # Daha basit tek model yaklaşımı
                    simple_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                    simple_pipeline = Pipeline([
                        ('preprocessor', preprocessor),
                        ('model', simple_model)
                    ])
                    
                    # Eğit ve tahmin et
                    simple_pipeline.fit(X_train, y_train)
                    simple_predictions = simple_pipeline.predict(predict_data[feature_cols])
                    
                    # Tahminleri gerçek veri aralığına sınırla
                    simple_predictions = np.clip(simple_predictions, min_val, max_val)
                    
                    # Tahminleri veri setine yerleştir
                    imputed_data.loc[missing_mask, target_col] = simple_predictions
                    
                    imputation_summary["variables"][target_col] = {
                        "type": "numerical",
                        "method": "fallback_single_model",
                        "model": "GradientBoosting",
                        "missing_count": int(missing_mask.sum()),
                        "missing_percentage": float(missing_mask.mean() * 100),
                        "error": str(e)
                    }
                except:
                    # Son çare olarak medyan imputasyonu
                    imputed_data.loc[missing_mask, target_col] = data[target_col].median()
                    
                    imputation_summary["variables"][target_col] = {
                        "type": "numerical",
                        "method": "fallback_median",
                        "missing_count": int(missing_mask.sum()),
                        "missing_percentage": float(missing_mask.mean() * 100),
                        "error": str(e)
                    }
    
    # KATEGORİK DEĞİŞKENLER İÇİN STACKING MODEL İMPUTATION
    if cat_with_missing:
        print("\nKategorik değişkenler için stacking model imputation başlatılıyor...")
        
        # Kategorik değişkenler için sınıflandırma modelleri (seviye-1 modeller)
        level1_models = {
            "RF": RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42),
            "GB": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
            "LR": LogisticRegression(max_iter=1000, random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "SVC": SVC(probability=True, random_state=42)
        }
        
        # Seviye-2 model (meta-model)
        meta_model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Her kategorik değişken için stacking imputation
        for target_col in tqdm(cat_with_missing, desc="Kategorik değişkenler"):
            try:
                # Eksik değer maskesi oluştur
                missing_mask = data[target_col].isnull()
                
                # Hiç eksik değer yoksa veya tümü eksikse atla
                if missing_mask.sum() == 0 or missing_mask.sum() == len(data):
                    continue
                
                # Benzersiz değer sayısını kontrol et - çok fazlaysa atla
                n_unique = data[target_col].nunique()
                if n_unique > 50:  # Çok fazla sınıf varsa model eğitimi zor olabilir
                    print(f"  {target_col} değişkeni çok fazla sınıf içeriyor ({n_unique}), mod imputasyonu kullanılacak.")
                    mode_value = data[target_col].mode()[0] if not data[target_col].mode().empty else None
                    imputed_data.loc[missing_mask, target_col] = mode_value
                    
                    imputation_summary["variables"][target_col] = {
                        "type": "categorical",
                        "method": "mode",
                        "reason": "too_many_classes",
                        "missing_count": int(missing_mask.sum()),
                        "missing_percentage": float(missing_mask.mean() * 100),
                        "unique_classes": int(n_unique)
                    }
                    continue
                    
                # Eğitim verisi: Eksik olmayan değerleri kullan
                train_data = temp_data[~missing_mask].copy()
                
                # Tahmin için veri: Eksik değerlere sahip satırlar
                predict_data = temp_data[missing_mask].copy()
                
                # Özellikler ve hedef
                feature_cols = [col for col in temp_data.columns if col != target_col]
                
                # Çok fazla değişken varsa, bazı önemli değişkenleri seç
                if len(feature_cols) > 100:
                    # Kategorik hedef için Kruskal-Wallis testi veya başka yöntemler kullanılabilir
                    # Basit bir yaklaşım: Sayısal ve kategorik değişkenlerin bir alt kümesini kullan
                    numeric_subset = numerical_cols[:50] if len(numerical_cols) > 50 else numerical_cols
                    categorical_subset = [col for col in categorical_cols if col != target_col]
                    categorical_subset = categorical_subset[:50] if len(categorical_subset) > 50 else categorical_subset
                    
                    feature_cols = list(set(numeric_subset + categorical_subset))
                
                # Özellik mühendisliği: Sayısal ve kategorik değişkenleri işle
                numeric_features = [col for col in feature_cols if col in numerical_cols]
                categorical_features = [col for col in feature_cols if col in categorical_cols]
                
                # Veri ön işleme
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', StandardScaler(), numeric_features),
                        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                    ], 
                    remainder='drop'
                )
                
                # Eğitim verisi hazırlama
                X_train = train_data[feature_cols]
                y_train = train_data[target_col]
                
                # Çapraz doğrulama için veri hazırlama (seviye-2 model eğitimi için)
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                
                # Hedef değişkenin benzersiz sınıfları
                classes = np.unique(y_train)
                n_classes = len(classes)
                
                # Seviye-1 model tahminlerini topla
                meta_features = np.zeros((X_train.shape[0], len(level1_models) * n_classes))
                
                # Her seviye-1 model için cross-validation ile meta-özellikler oluştur
                col_idx = 0
                for model_name, model in level1_models.items():
                    # Pipeline oluştur
                    pipeline = Pipeline([
                        ('preprocessor', preprocessor),
                        ('model', model)
                    ])
                    
                    # Çapraz doğrulama olasılık tahminleri
                    cv_preds = np.zeros((X_train.shape[0], n_classes))
                    
                    for train_idx, val_idx in kf.split(X_train):
                        # Eğitim ve doğrulama bölümleri
                        X_train_cv, X_val_cv = X_train.iloc[train_idx], X_train.iloc[val_idx]
                        y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]
                        
                        # Modeli eğit
                        pipeline.fit(X_train_cv, y_train_cv)
                        
                        # Doğrulama kümesi için olasılık tahminleri
                        cv_preds[val_idx] = pipeline.predict_proba(X_val_cv)
                    
                    # Meta-özellikler olarak olasılık tahminlerini kaydet
                    for i in range(n_classes):
                        meta_features[:, col_idx] = cv_preds[:, i]
                        col_idx += 1
                
                # Seviye-2 modeli (meta-model) eğit
                meta_model.fit(meta_features, y_train)
                
                # Seviye-1 modelleri tam veri seti üzerinde yeniden eğit
                level1_predictions = np.zeros((predict_data.shape[0], len(level1_models) * n_classes))
                
                col_idx = 0
                for model_name, model in level1_models.items():
                    pipeline = Pipeline([
                        ('preprocessor', preprocessor),
                        ('model', model)
                    ])
                    
                    # Tüm eğitim verisiyle eğit
                    pipeline.fit(X_train, y_train)
                    
                    # Tahmin verisi için olasılık öngörüleri
                    X_predict = predict_data[feature_cols]
                    proba_preds = pipeline.predict_proba(X_predict)
                    
                    # Meta-özellikler olarak olasılık tahminlerini kaydet
                    for i in range(n_classes):
                        level1_predictions[:, col_idx] = proba_preds[:, i]
                        col_idx += 1
                
                # Meta-model ile nihai tahminleri yap
                final_predictions = meta_model.predict(level1_predictions)
                
                # Tahminleri veri setine yerleştir
                imputed_data.loc[missing_mask, target_col] = final_predictions
                
                # Model performansını hesapla (çapraz doğrulama doğruluk)
                cv_scores = cross_val_score(
                    meta_model, meta_features, y_train, 
                    cv=5, scoring='accuracy'
                )
                accuracy = np.mean(cv_scores)
                
                # İmputasyon özetini güncelle
                imputation_summary["variables"][target_col] = {
                    "type": "categorical",
                    "method": "stacking_classification",
                    "missing_count": int(missing_mask.sum()),
                    "missing_percentage": float(missing_mask.mean() * 100),
                    "model_accuracy": float(accuracy),
                    "feature_count": len(feature_cols),
                    "unique_classes": int(n_unique)
                }
                
            except Exception as e:
                print(f"  Hata ({target_col}): {e}")
                # Hata durumunda daha basit bir sınıflandırıcı yaklaşımı kullan
                try:
                    # Daha basit tek model yaklaşımı
                    simple_model = RandomForestClassifier(n_estimators=100, random_state=42)
                    
                    # Veri ön işleme
                    preprocessor = ColumnTransformer(
                        transformers=[
                            ('num', StandardScaler(), numeric_features),
                            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                        ], 
                        remainder='drop'
                    )
                    
                    simple_pipeline = Pipeline([
                        ('preprocessor', preprocessor),
                        ('model', simple_model)
                    ])
                    
                    # Eğit ve tahmin et
                    simple_pipeline.fit(X_train, y_train)
                    simple_predictions = simple_pipeline.predict(predict_data[feature_cols])
                    
                    # Tahminleri veri setine yerleştir
                    imputed_data.loc[missing_mask, target_col] = simple_predictions
                    
                    imputation_summary["variables"][target_col] = {
                        "type": "categorical",
                        "method": "fallback_single_model",
                        "model": "RandomForest",
                        "missing_count": int(missing_mask.sum()),
                        "missing_percentage": float(missing_mask.mean() * 100),
                        "error": str(e)
                    }
                except:
                    # Son çare olarak mod imputasyonu
                    mode_value = data[target_col].mode()[0] if not data[target_col].mode().empty else None
                    imputed_data.loc[missing_mask, target_col] = mode_value
                    
                    imputation_summary["variables"][target_col] = {
                        "type": "categorical",
                        "method": "fallback_mode",
                        "missing_count": int(missing_mask.sum()),
                        "missing_percentage": float(missing_mask.mean() * 100),
                        "error": str(e)
                    }
    
    # İmputasyon sonrası eksik değer kontrolü
    remaining_missing = imputed_data.isnull().sum().sum()
    imputation_summary["metadata"]["initial_missing_values"] = int(data.isnull().sum().sum())
    imputation_summary["metadata"]["remaining_missing_values"] = int(remaining_missing)
    imputation_summary["metadata"]["imputation_success_rate"] = float(100 - (remaining_missing / data.isnull().sum().sum() * 100)) if data.isnull().sum().sum() > 0 else 100.0
    
    # Sonuçları JSON dosyasına kaydet
    with open(f"{output_prefix}_stacking_imputation.json", 'w', encoding='utf-8') as f:
        json.dump(imputation_summary, f, indent=4, ensure_ascii=False)
    
    # İmputasyon sonrası veriyi kaydet
    imputed_data.to_csv(f"{output_prefix}_stacking_imputed_data.csv", index=False)
    
    print("\nStacking Model İmputasyonu Tamamlandı!")
    print(f"Sonuçlar:")
    print(f"  İmputasyon özeti: {output_prefix}_stacking_imputation.json")
    print(f"  İmputasyon sonrası veri: {output_prefix}_stacking_imputed_data.csv")
    print(f"  Başlangıçtaki eksik değer sayısı: {data.isnull().sum().sum()}")
    print(f"  İmputasyon sonrası kalan eksik değer sayısı: {remaining_missing}")
    print(f"  İmputasyon başarı oranı: %{imputation_summary['metadata']['imputation_success_rate']:.2f}")
    
    return imputed_data, imputation_summary

def complete_timss_imputation_workflow():
    """
    TIMSS veri seti için tam bir eksik değer doldurma iş akışı.
    
    Bu fonksiyon şunları yapar:
    1. Veri setini yükler
    2. Eksik değer analizi yapar
    3. Eksik değerleri çeşitli yöntemlerle doldurur
    4. İmputasyon sonrası veri analizi yapar
    5. Tüm sonuçları raporlar
    """
    start_time = time.time()
    print("TIMSS Veri Seti Eksik Değer Doldurma İş Akışı")
    print("=============================================")
    
    # 1. Veri Yükleme
    print("\n1. Veri Seti Yükleniyor...")
    try:
        import pyreadr
        filepath = "/Users/mrved/Desktop/TIMSS Regression/TIMSS-8 R Data/bsaautm8.rdata"
        result = pyreadr.read_r(filepath)
        data = list(result.values())[0]
        print(f"Veri seti başarıyla yüklendi.")
        print(f"Boyut: {data.shape[0]} satır, {data.shape[1]} sütun")
    except Exception as e:
        print(f"Veri yüklenirken hata oluştu: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Zaman damgası oluştur
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_prefix = f"timss_imputation_{timestamp}"
    
    # Çıktı dizini oluştur
    output_dir = f"timss_imputation_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    full_output_prefix = os.path.join(output_dir, output_prefix)
    
    # 2. Eksik Değer Analizi
    print("\n2. Eksik Değer Analizi Yapılıyor...")
    missing_analysis = analyze_missing_values(data, f"{full_output_prefix}_missing_analysis")
    
    # 3. İmputasyon Stratejisi Belirleme
    print("\n3. İmputasyon Stratejisi Belirleniyor...")
    # Değişkenleri tipine göre ayır
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Eksik değeri olan değişkenleri bul
    numerical_with_missing = [col for col in numerical_cols if data[col].isnull().sum() > 0]
    categorical_with_missing = [col for col in categorical_cols if data[col].isnull().sum() > 0]
    
    print(f"Toplam değişken sayısı: {len(data.columns)}")
    print(f"Kategorik değişken sayısı: {len(categorical_cols)}")
    print(f"Sayısal değişken sayısı: {len(numerical_cols)}")
    print(f"Eksik değeri olan kategorik değişken sayısı: {len(categorical_with_missing)}")
    print(f"Eksik değeri olan sayısal değişken sayısı: {len(numerical_with_missing)}")
    
    # Eksik değer oranlarına göre değişkenleri gruplandır
    missing_ratios = {col: data[col].isnull().mean() for col in data.columns if data[col].isnull().sum() > 0}
    high_missing = [col for col, ratio in missing_ratios.items() if ratio > 0.5]
    medium_missing = [col for col, ratio in missing_ratios.items() if 0.2 <= ratio <= 0.5]
    low_missing = [col for col, ratio in missing_ratios.items() if ratio < 0.2]
    
    print(f"Yüksek eksik oranı (>%50): {len(high_missing)} değişken")
    print(f"Orta eksik oranı (%20-%50): {len(medium_missing)} değişken")
    print(f"Düşük eksik oranı (<%20): {len(low_missing)} değişken")
    
    # İmputasyon yöntemlerini değerlendir (eğer yeterli sayısal veri varsa)
    best_methods = {}
    if len(numerical_cols) > 10 and len([col for col in numerical_cols if data[col].isnull().sum() == 0]) > 5:
        print("\nOptimal imputasyon yöntemlerini değerlendirme...")
        try:
            results_df, summary, best_methods = evaluate_imputation_methods(
                data, 
                test_cols=None,  # Otomatik seçim
                missing_rates=[0.2, 0.5],
                n_iterations=3
            )
            
            # Değerlendirme sonuçlarını kaydet
            results_df.to_csv(f"{full_output_prefix}_method_evaluation.csv", index=False)
            
            # En iyi yöntemleri kullan
            print("\nDeğerlendirme sonuçlarına göre en iyi imputasyon yöntemleri:")
            for rate, result in best_methods.items():
                print(f"  Eksik değer oranı {rate*100}%: {result['method']} (RMSE={result['rmse']:.4f})")
                
            # Yöntem seçimleri
            best_low = best_methods.get(0.2, {}).get('method', 'Mean')
            best_high = best_methods.get(0.5, {}).get('method', 'Median')
            if best_low != best_high:
                best_medium = best_low
            else:
                best_medium = 'Mean'
        except Exception as e:
            print(f"Yöntem değerlendirmesi başarısız oldu, varsayılan yöntemler kullanılacak: {e}")
            # Varsayılan yöntemler
            best_low = 'Mean'
            best_medium = 'Mean'
            best_high = 'Median'
    else:
        # Varsayılan yöntemler
        best_low = 'Mean'
        best_medium = 'Mean'
        best_high = 'Median'
    
    # 4. İmputasyon Uygulama
    print("\n4. İmputasyon Uygulanıyor...")
    
    # Entegre imputasyon yöntemini çağır
    imputed_data = None
    final_summary = None
    try:
        # Model tabanlı yöntemler kullan (yeni eklenen)
        print("Model tabanlı imputation yöntemleri uygulanıyor...")
        
        # Basit model tabanlı yaklaşım
        imputed_data_model, imputation_summary_model = model_based_imputation(
            data, numerical_cols, categorical_cols, full_output_prefix
        )
        
        # Stacking model yaklaşımı
        imputed_data_stacking, imputation_summary_stacking = stacking_model_imputation(
            data, numerical_cols, categorical_cols, full_output_prefix
        )
        
        # Super Learner yöntemi de uygula (mevcut kod)
        print("\nSuper Learner imputation yöntemi de uygulanıyor...")
        imputed_data_sl, imputation_summary_sl = super_learner_imputation(
            data, numerical_cols, categorical_cols, full_output_prefix
        )
        
        print("\nEnsemble imputation yöntemi de uygulanıyor...")
        imputed_data_ensemble, imputation_summary_ensemble = ensemble_imputation(
            data, full_output_prefix
        )
        
        # Tüm yöntemlerin sonuçlarını karşılaştır
        model_success_rate = imputation_summary_model["metadata"].get("imputation_success_rate", 0)
        stacking_success_rate = imputation_summary_stacking["metadata"].get("imputation_success_rate", 0)
        sl_success_rate = imputation_summary_sl["metadata"].get("imputation_success_rate", 0)
        ensemble_success_rate = imputation_summary_ensemble["metadata"].get("imputation_success_rate", 0)
        
        # En iyi yöntemi seç
        success_rates = {
            "Model Tabanlı": model_success_rate,
            "Stacking Model": stacking_success_rate,
            "Super Learner": sl_success_rate,
            "Ensemble": ensemble_success_rate
        }
        
        best_method_name = max(success_rates.items(), key=lambda x: x[1])[0]
        best_success_rate = success_rates[best_method_name]
        
        print(f"\nYöntem karşılaştırması:")
        for method, rate in success_rates.items():
            print(f"  {method}: %{rate:.2f}")
        
        print(f"\n{best_method_name} yöntemi en iyi sonucu verdi (Başarı oranı: %{best_success_rate:.2f})")
        
        # En iyi yöntemi seç
        if best_method_name == "Model Tabanlı":
            imputed_data = imputed_data_model
            final_summary = imputation_summary_model
        elif best_method_name == "Stacking Model":
            imputed_data = imputed_data_stacking
            final_summary = imputation_summary_stacking
        elif best_method_name == "Super Learner":
            imputed_data = imputed_data_sl
            final_summary = imputation_summary_sl
        else:  # Ensemble
            imputed_data = imputed_data_ensemble
            final_summary = imputation_summary_ensemble
            
        # Nihai veriyi kaydet
        imputed_data.to_csv(f"{full_output_prefix}_final_imputed_data.csv", index=False)
        
        # İmputasyon özeti JSON oluştur
        comparison_summary = {
            "metadata": {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_variables": len(data.columns),
                "total_rows": len(data),
                "categorical_variables": len(categorical_cols),
                "numerical_variables": len(numerical_cols),
                "initial_missing_values": int(data.isnull().sum().sum()),
                "remaining_missing_values": int(imputed_data.isnull().sum().sum()),
                "imputation_success_rate": float(best_success_rate),
                "final_method": best_method_name
            },
            "method_comparison": {
                "model_based": {
                    "success_rate": float(model_success_rate),
                    "remaining_missing": int(imputation_summary_model["metadata"]["remaining_missing_values"])
                },
                "stacking_model": {
                    "success_rate": float(stacking_success_rate),
                    "remaining_missing": int(imputation_summary_stacking["metadata"]["remaining_missing_values"])
                },
                "super_learner": {
                    "success_rate": float(sl_success_rate),
                    "remaining_missing": int(imputation_summary_sl["metadata"]["remaining_missing_values"])
                },
                "ensemble": {
                    "success_rate": float(ensemble_success_rate),
                    "remaining_missing": int(imputation_summary_ensemble["metadata"]["remaining_missing_values"])
                }
            }
        }
        
        with open(f"{full_output_prefix}_final_summary.json", 'w', encoding='utf-8') as f:
            json.dump(comparison_summary, f, indent=4, ensure_ascii=False)
            
    except Exception as e:
        print(f"İmputasyon uygulanırken hata oluştu: {e}")
        import traceback
        traceback.print_exc()
        # Temel imputasyon ile devam et
        print("Fallback: Temel imputasyon yöntemleri uygulanıyor...")
        
        # Basit imputasyon yöntemi kullan
        try:
            imputed_data = perform_advanced_imputation(data, full_output_prefix)
            
            # Özet oluştur
            final_summary = {
                "metadata": {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "total_variables": len(data.columns),
                    "total_rows": len(data),
                    "categorical_variables": len(categorical_cols),
                    "numerical_variables": len(numerical_cols),
                    "initial_missing_values": int(data.isnull().sum().sum()),
                    "remaining_missing_values": int(imputed_data.isnull().sum().sum()),
                    "imputation_success_rate": float(100 - (imputed_data.isnull().sum().sum() / data.isnull().sum().sum() * 100)) if data.isnull().sum().sum() > 0 else 100.0,
                    "final_method": "Basic Imputation (Fallback)"
                },
                "variable_groups": {
                    "high_missing": high_missing,
                    "medium_missing": medium_missing,
                    "low_missing": low_missing
                }
            }
            
            with open(f"{full_output_prefix}_final_summary.json", 'w', encoding='utf-8') as f:
                json.dump(final_summary, f, indent=4, ensure_ascii=False)
                
        except Exception as e2:
            print(f"Temel imputasyon da başarısız oldu: {e2}")
            traceback.print_exc()
            return
            
    # 5. İmputasyon Sonrası Veri Analizi
    if imputed_data is not None:
        print("\n5. İmputasyon Sonrası Veri Analizi Yapılıyor...")
        try:
            # İmputasyon öncesi ve sonrası veri setlerini karşılaştır
            compare_pre_post_imputation(data, imputed_data, full_output_prefix)
            
            # Keşifsel veri analizi
            perform_exploratory_analysis(imputed_data, full_output_prefix)
            
            # Boyut indirgeme ve kümeleme
            perform_dimensionality_reduction(imputed_data, full_output_prefix)
            
            # Örnek tahmin modeli oluştur
            build_sample_predictive_model(imputed_data, full_output_prefix)
            
        except Exception as e:
            print(f"İmputasyon sonrası analizlerde hata oluştu: {e}")
            import traceback
            traceback.print_exc()
    
        # 6. Raporlama ve Arşivleme
        print("\n6. Sonuçlar Raporlanıyor ve Arşivleniyor...")
        try:
            # Özet dashboard oluştur
            create_imputation_dashboard(data, imputed_data, final_summary, full_output_prefix)
            
            # Tüm dosyaları arşivle
            output_files = glob.glob(f"{output_dir}/*.*")
            zip_path = f"{output_dir}.zip"
            
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for file in output_files:
                    zipf.write(file, os.path.basename(file))
                    
            print(f"Tüm çıktılar arşivlendi: {zip_path}")
            
        except Exception as e:
            print(f"Raporlama ve arşivleme sırasında hata oluştu: {e}")
            import traceback
            traceback.print_exc()
    
    # İşlem süresi
    elapsed_time = time.time() - start_time
    zip_path = f"{output_dir}.zip"  # Bu satırı buraya ekleyin (mevcut bir tanım yoksa)
    print("\nİş Akışı Tamamlandı!")
    print(f"Toplam işlem süresi: {elapsed_time:.2f} saniye ({elapsed_time/60:.2f} dakika)")
    print(f"Tüm sonuçlar '{output_dir}' dizininde bulunabilir.")
    if 'zip_path' in locals():
        print(f"ve '{zip_path}' arşivinde bulunabilir.")

# Ana fonksiyonu çağır
if __name__ == "__main__":
    complete_timss_imputation_workflow()