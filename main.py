import pandas as pd
import librosa
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import correlate
from sklearn.metrics import confusion_matrix

# --- 1. VERİ SETİ YÖNETİMİ (Madde 16-18) ---
BASE_DIR = r"C:\Users\Dilan\OneDrive - ISTANBUL SAGLIK VE TEKNOLOJI UNIVERSITESI\Masaüstü\speech_project\Dataset"

def standardize_label(label):
    """Etiketleri Male, Female, Child olarak standartlaştırır"""
    label = str(label).lower().strip()
    if any(x in label for x in ['man', 'erkek', 'male', 'm']): return 'Male'
    if any(x in label for x in ['wom', 'kad', 'female', 'f']): return 'Female'
    if any(x in label for x in ['child', 'cocuk', 'çocuk', 'c']): return 'Child'
    return None

# --- 2. OTOKORELASYON VE FFT ARAŞTIRMASI (Madde 3.A) ---
def plot_research_comparison(y, sr, fname):
    """FFT ve Otokorelasyonu rapor için yan yana çizer (Madde 36, 62)"""
    frame_size = int(0.03 * sr) # 30ms durağan pencere (Madde 23)
    frame = y[int(sr):int(sr + frame_size)] 

    plt.figure(figsize=(12, 5))
    # FFT Spektrumu
    plt.subplot(1, 2, 1)
    fft_spec = np.abs(np.fft.rfft(frame))
    freqs = np.fft.rfftfreq(len(frame), 1/sr)
    plt.plot(freqs[:150], fft_spec[:150], color='blue')
    plt.title(f"FFT Spectrum (Frequency Domain) - {fname}")
    plt.xlabel("Frequency (Hz)")

    # Otokorelasyon (Madde 29: Rτ = Σ x[n]x[n-τ])
    plt.subplot(1, 2, 2)
    autocorr = correlate(frame, frame, mode='full')[len(frame)-1:]
    plt.plot(autocorr[:500], color='red')
    plt.title(f"Autocorrelation (Time Domain) - {fname}")
    plt.xlabel("Lag (Samples)")
    
    plt.tight_layout()
    plt.show() # Bu grafiği raporunuza ekleyin

# --- 3. ÖZNİTELİK ÇIKARIMI (Madde 3.1 & 3.2) ---
def compute_autocorr_f0(y, sr):
    """Zaman alanında otokorelasyon ile F0 hesabı (Madde 28, 32)"""
    r_min, r_max = int(sr / 450), int(sr / 70) # 70-450 Hz aralığı
    f_len, h_len = int(0.025 * sr), int(0.0125 * sr) # 25ms pencere
    
    energy = librosa.feature.rms(y=y, frame_length=f_len, hop_length=h_len)[0]
    voiced_mask = energy > (np.mean(energy) * 0.5) # Sesli bölge tespiti (Madde 24)
    
    f0_values = []
    frames = librosa.util.frame(y, frame_length=f_len, hop_length=h_len)
    for i in range(min(frames.shape[1], len(voiced_mask))):
        if voiced_mask[i]:
            frame_data = frames[:, i]
            autocorr = correlate(frame_data, frame_data, mode='full')[len(frame_data)-1:]
            if len(autocorr) > r_max:
                peak = np.argmax(autocorr[r_min:r_max]) + r_min
                f0_values.append(sr / peak)
    return np.mean(f0_values) if f0_values else 0

# --- 4. ANA DÖNGÜ VE SINIFLANDIRMA (Madde 4) ---
excel_files = glob.glob(os.path.join(BASE_DIR, "**", "*.xlsx"), recursive=True)
all_results = []
comparison_done = False

print("--- 🎧 Analiz Başlatıldı ---")
for excel in excel_files:
    df = pd.read_excel(excel)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    p_col = next((c for c in df.columns if any(x in c for x in ['path', 'file', 'dosya'])), df.columns[0])
    g_col = next((c for c in df.columns if any(x in c for x in ['gender', 'cinsiyet', 'etiket'])), None)

    for _, row in df.iterrows():
        actual = standardize_label(row[g_col])
        if not actual: continue
        
        fname = os.path.basename(str(row[p_col]).replace('\\', '/'))
        if not fname.lower().endswith('.wav'): fname += '.wav'
        
        fpath = glob.glob(os.path.join(BASE_DIR, "**", fname), recursive=True)
        if fpath:
            y, sr = librosa.load(fpath[0], sr=None)
            if not comparison_done:
                plot_research_comparison(y, sr, fname)
                comparison_done = True
            
            f0 = compute_autocorr_f0(y, sr)
            if f0 > 0:
                # Kural Tabanlı Tahmin (Madde 10)
                pred = "Male" if f0 < 165 else ("Female" if f0 < 255 else "Child")
                all_results.append({'File': fname, 'Actual': actual, 'Predicted': pred, 'F0': f0})

# --- 5. RAPORLAMA VE PERFORMANS (Madde 5 & 64) ---
if all_results:
    res_df = pd.DataFrame(all_results)
    res_df['Success'] = (res_df['Actual'] == res_df['Predicted'])
    
    print("\n" + "="*60 + "\nPROJE İSTATİSTİK TABLOSU (Madde 5)\n" + "="*60)
    stats = res_df.groupby('Actual').agg(
        Number_of_Samples=('F0', 'count'),
        Average_F0_Hz=('F0', 'mean'),
        Standard_Deviation=('F0', 'std'),
        Success_Rate_Percent=('Success', lambda x: x.mean() * 100)
    ).reset_index()
    print(stats.to_string(index=False))

    # Confusion Matrix (Madde 64)
    plt.figure(figsize=(8, 6))
    labels = ["Male", "Female", "Child"]
    cm = confusion_matrix(res_df['Actual'], res_df['Predicted'], labels=labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.show()

    res_df.to_excel("DILAN_SPEECH_FINAL_REPORT.xlsx", index=False)

import gradio as gr

def live_classification_interface(audio_file):
    """
    Demo sırasında yüklenen veya kaydedilen sesi analiz eder.
    """
    if audio_file is None:
        return "Lütfen bir ses dosyası yükleyin veya kaydedin."
    
    # 1. Sesi yükle
    y, sr = librosa.load(audio_file, sr=None)
    
    # 2. Senin yazdığın Otokorelasyon fonksiyonunu kullanarak F0 hesapla
    f0 = compute_autocorr_f0(y, sr)
    
    # 3. Kural Tabanlı Karar Mekanizman (Yönergedeki eşik değerlerin)
    if f0 < 165:
        prediction = "ERKEK (Male) 👨"
        color = "Mavi"
    elif f0 < 255:
        prediction = "KADIN (Female) 👩"
        color = "Pembe"
    else:
        prediction = "ÇOCUK (Child) 🧒"
        color = "Yeşil"
        
    return f"Hesaplanan Temel Frekans (F0): {f0:.2f} Hz\nTahmin Edilen Sınıf: {prediction}"

import gradio as gr

# --- DEMO ARAYÜZÜ FONKSİYONU ---
def demo_analiz(ses_yolu):
    if ses_yolu is None:
        return "Lütfen bir ses yükleyin veya mikrofona konuşun."
    
    # 1. Sesi yükle (Kendi kütüphanelerini kullanır)
    y, sr = librosa.load(ses_yolu, sr=None)
    
    # 2. Senin yazdığın F0 hesaplama fonksiyonunu çağırıyoruz
    f0 = compute_autocorr_f0(y, sr)
    
    # 3. Kural tabanlı sınıflandırma (Slaytında anlatacağın mantık)
    if f0 < 165:
        tahmin = "ERKEK (Male) 👨"
    elif f0 < 255:
        tahmin = "KADIN (Female) 👩"
    else:
        tahmin = "ÇOCUK (Child) 🧒"
        
    return f"Analiz Sonucu:\nHesaplanan F0: {f0:.2f} Hz\nTahmin: {tahmin}"

# --- ARAYÜZ TASARIMI ---
with gr.Blocks(title="COE216 Ses Analizi Demosu") as demo:
    gr.Markdown("# 🎙️ Ses Analizi ile Cinsiyet Sınıflandırma")
    gr.Markdown("Bu arayüz, Otokorelasyon (Autocorrelation) yöntemiyle F0 değerini hesaplar.")
    
    with gr.Row():
        ses_girisi = gr.Audio(type="filepath", label="Ses Kaydet veya Dosya Yükle")
        sonuc_alani = gr.Textbox(label="Sınıflandırma Sonucu", interactive=False)
    
    btn = gr.Button("Analiz Et", variant="primary")
    btn.click(fn=demo_analiz, inputs=ses_girisi, outputs=sonuc_alani)

# --- ARAYÜZÜ BAŞLAT ---
if __name__ == "__main__":
    print("\n" + "="*30)
    print("DEMO BAŞLATILIYOR...")
    print("Aşağıdaki linke tıkla veya tarayıcına yapıştır:")
    print("="*30)
    demo.launch(inbrowser=True) # Bu komut tarayıcıyı otomatik açar
