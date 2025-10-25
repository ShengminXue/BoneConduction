import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, ifft
import os
from random import uniform

def load_audio(file_path):
    """加载WAV音频文件"""
    sample_rate, data = wavfile.read(file_path)
    return sample_rate, data

def time_to_frequency_domain(time_signal):
    """时域信号转换为频域"""
    return fft(time_signal)

def frequency_to_time_domain(freq_signal):
    """频域信号转换回时域"""
    return ifft(freq_signal).real

def butter(freqs, cutoff, order=4, btype='bandpass', amplify=1, shift=1):
    """
    生成Butterworth滤波器的频域响应
    :param freqs: 频率数组 (Hz)
    :param cutoff: 截止频率 (低/高通) 或 [low, high] (带通/带阻)
    :param order: 滤波器阶数
    :param btype: 滤波器类型 ('lowpass', 'highpass', 'bandpass', 'bandstop')
    :return: Butterworth滤波器频域响应
    """
    # 初始化滤波器响应
    H = np.ones_like(freqs, dtype=np.float64)
    w = 2 * np.pi * np.abs(freqs)  # 角频率 (rad/s)
    
    # 处理不同滤波器类型
    if btype == 'lowpass':
        # 低通滤波器
        if isinstance(cutoff, (list, tuple, np.ndarray)):
            cutoff = cutoff[1] if len(cutoff) > 1 else cutoff[0]
        w_c = 2 * np.pi * cutoff
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = w / w_c
            H = (amplify-1) / np.sqrt(1 + (ratio) ** (2 * order))+shift
    
    elif btype == 'highpass':
        # 高通滤波器
        if isinstance(cutoff, (list, tuple, np.ndarray)):
            cutoff = cutoff[0] if len(cutoff) > 0 else cutoff
        w_c = 2 * np.pi * cutoff
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = w_c / w
            H = (amplify-1) / np.sqrt(1 + (ratio) ** (2 * order))+shift
    
    elif btype == 'bandpass':
        H = np.ones_like(freqs, dtype=np.float64)
        w = 2 * np.pi * np.abs(freqs)  # 角频率 (rad/s)

        # 带通滤波器
        low, high = cutoff
        w_low = 2 * np.pi * low
        w_high = 2 * np.pi * high
        w0 = np.sqrt(w_low * w_high)  # 中心频率
        bandwidth = w_high - w_low    # 带宽
    
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = (w**2 - w0**2) / (w * bandwidth)
            # 带通滤波器
            H = (amplify - 1) / np.sqrt(1 + (ratio) ** (2 * order)) + shift
        
            # 处理频率为零的情况
            H[np.isnan(H)] = shift
    
        return H
    
    elif btype == 'bandstop':
        # 带阻滤波器
        if not isinstance(cutoff, (list, tuple, np.ndarray)) or len(cutoff) < 2:
            raise ValueError("带阻滤波器需要两个截止频率 [low, high]")
        
        low, high = cutoff
        w_low = 2 * np.pi * low
        w_high = 2 * np.pi * high
        w0 = np.sqrt(w_low * w_high)  # 中心频率
        bandwidth = w_high - w_low    # 带宽
        
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = (w * bandwidth) / (w**2 - w0**2)
            H = (amplify-1) / np.sqrt(1 + (ratio) ** (2 * order)) + shift
    
    else:
        raise ValueError(f"未知滤波器类型: {btype}")
    
    # 处理特殊值
    H[np.isnan(H)] = 0
    H[np.isinf(H)] = 0
    
    # 确保对称性（实数信号的FFT是共轭对称的）
    if len(H) % 2 == 0:
        # 偶数长度
        H[len(H)//2+1:] = H[len(H)//2-1:0:-1]
    else:
        # 奇数长度
        H[(len(H)+1)//2:] = H[(len(H)-1)//2:0:-1]
    
    return H

def apply_frequency_filter(freq_signal, sample_rate, btype='bandpass', cutoff=None, order=4, amplify = 1, shift = 1):
    """
    在频域应用Butterworth滤波器
    :param freq_signal: 频域信号
    :param sample_rate: 采样率
    :param btype: 滤波器类型 ('lowpass', 'highpass', 'bandpass', 'bandstop')
    :param cutoff: 截止频率 (标量或[low, high])
    :param order: 滤波器阶数
    :return: 滤波后的频域信号
    """
    # 设置默认截止频率
    if cutoff is None:
        if btype in ['lowpass', 'highpass']:
            cutoff = sample_rate / 4  # 默认四分之一采样率
        else:
            cutoff = [sample_rate/8, sample_rate/3]  # 默认1/8到1/3采样率
    
    # 创建频率数组
    n = len(freq_signal)
    freqs = np.fft.fftfreq(n, d=1/sample_rate)
    
    # 生成滤波器响应
    H = butter(freqs, cutoff, order, btype, amplify, shift)
    
    # 应用滤波器
    filtered_freq = freq_signal * H
    
    return filtered_freq

def plot_signals(original_time, original_freq, 
                processed_time, processed_freq, 
                sample_rate, save_path='output_plots.png'):
    """
    绘制并保存时域和频域对比图（幅度显示）
    :param original_time: 原始时域信号
    :param original_freq: 原始频域信号
    :param processed_time: 处理后时域信号
    :param processed_freq: 处理后频域信号
    :param sample_rate: 采样率
    :param save_path: 图像保存路径
    """
    plt.figure(figsize=(15, 10))
    
    # 原始时域
    plt.subplot(2, 2, 1)
    time_axis = np.arange(len(original_time)) / sample_rate
    plt.plot(time_axis, original_time)
    plt.title('Original Signal - Time Domain')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    # 原始频域（幅度）
    plt.subplot(2, 2, 2)
    freq_axis = np.linspace(0, sample_rate, len(original_freq))
    magnitude = np.abs(original_freq)  # 直接取幅度
    plt.plot(freq_axis, magnitude)
    plt.title('Original Signal - Frequency Domain (Amplitude)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.xlim(0, sample_rate/2)
    plt.grid(True, alpha=0.3)
    
    # 处理后时域
    plt.subplot(2, 2, 3)
    time_axis = np.arange(len(processed_time)) / sample_rate
    plt.plot(time_axis, processed_time)
    plt.title('Processed Signal - Time Domain')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    # 处理后频域（幅度）
    plt.subplot(2, 2, 4)
    freq_axis = np.linspace(0, sample_rate, len(processed_freq))
    magnitude = np.abs(processed_freq)  # 直接取幅度
    plt.plot(freq_axis, magnitude)
    plt.title('Processed Signal - Frequency Domain (Amplitude)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.xlim(0, sample_rate/2)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_audio(data, sample_rate, output_path):
    """保存音频文件并确保目录存在"""
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 确保数据在int16范围内
    data = np.int16(data / np.max(np.abs(data)) * 32767)
    wavfile.write(output_path, sample_rate, data)
    print('save' + output_path)

def process_audio(input_file, sample_rate, btype='bandpass', cutoff=None, order=4, plot_path='audio_analysis.png', amplify = 1, shift = 1):
    """
    主处理函数 - 使用频域Butterworth滤波
    :param input_file: 输入音频文件路径
    :param output_file: 输出音频文件路径
    :param btype: 滤波器类型 ('lowpass', 'highpass', 'bandpass', 'bandstop')
    :param cutoff: 截止频率 (标量或[low, high])
    :param order: 滤波器阶数
    :param plot_path: 分析图保存路径
    """
    # 1. 加载音频文件
    #sample_rate, audio_data = load_audio(input_file)
    
    # 如果是立体声，只取一个声道
    #if len(audio_data.shape) > 1:
    #    audio_data = audio_data[:, 0]
    
    # 2. 转换为频域
    freq_signal = time_to_frequency_domain(audio_data)
    
    # 3. 应用频域滤波器
    filtered_freq = apply_frequency_filter(freq_signal, sample_rate, btype, cutoff, order, amplify, shift)
    
    # 4. 转换回时域
    filtered_audio = frequency_to_time_domain(filtered_freq)

    return(filtered_audio, filtered_freq)
    
    # 5. 绘制并保存分析图
#    plot_signals(
#        audio_data, freq_signal,
#        filtered_audio, filtered_freq,
#        sample_rate, plot_path
#    )
    
    # 6. 输出处理后的音频
#    save_audio(filtered_audio, sample_rate, output_file)
    
#    print(f"处理完成! 音频已保存至: {output_file}")
#    print(f"分析图已保存至: {plot_path}")

# 使用示例
if __name__ == "__main__":
    input_audio = 'sound.wav' #replace with sound file
    output_audio = 'filtered_output.wav'
    plot_path ='audio_analysis.png'
    
    # 滤波器配置
    filter_type = 'bandpass'  # 可选: 'lowpass', 'highpass', 'bandpass', 'bandstop'
    filter_order = 4          # 滤波器阶数
    cutoff_freq = [1, 1000] # 截止频率 (低/高通: 标量, 带通/带阻: [low, high])

    # order, cutoff, amplify, shifting, filepath
    parameters = [
        [15, [100, 800], 0.631, 1,],
        [15, [800, 2000], 1.413, 1],
        [15, [2000, 4000], 2.512, 1],
        [15, [4000, 10000], 0.398, 1]
    ]

i = 0
while i < 10:

    sample_rate, audio_data = load_audio(input_audio)
    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]
    audio_data_1 = audio_data

    for parameter in parameters:    
        rand = uniform(0.667,1.5)
        audio_data, filtered_freq = process_audio(
            audio_data,
            sample_rate,
            btype = 'bandpass',
            cutoff = parameter[1],
            order = parameter[0],
            plot_path ='audio_analysis' + str(i) + '.png',
            amplify = parameter[2] * rand,
            shift = parameter[3]      
        )
        print(parameter[2] * rand)

    freq_signal = time_to_frequency_domain(audio_data_1)
    filtered_audio = audio_data
    plot_signals(
        audio_data_1, freq_signal,
        filtered_audio, filtered_freq,
        sample_rate, 'audio_analysis' + str(i) + '.png'
    )

    save_audio(filtered_audio, sample_rate, 'filtered_output' + str(i) + '.wav')
    i += 1
