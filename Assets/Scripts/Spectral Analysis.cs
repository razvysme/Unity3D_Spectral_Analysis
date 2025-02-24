using System;
using System.Threading.Tasks;
using UnityEngine;
using TMPro;
using MathNet.Numerics;
using MathNet.Numerics.IntegralTransforms;

// This script is an implementation of some of the spectral descriptors as found at https://se.mathworks.com/help/audio/ug/spectral-descriptors.html

[RequireComponent(typeof(AudioSource))]
public class AudioSpectrum : MonoBehaviour
{
    public class SpectralAnalysisResult
    {
        public string Label;
        public float SpectralCentroid;
        public float SpectralSpread;
        public float SpectralEntropy;
        public float RMSEnergy;
        public float ZeroCrossingRate;
        public float SpectralFlux;
        public float SpectralTilt;
        public float[] Spectrum; 
    }

    private AudioSource audioSource;
    public FFTWindow fftWindow;
    private readonly int analysWindowLengthPow = 4; // this sets the length of the analysis window as power of 2 * the input buffer size (1024 in my case) -> larger windows result in better accuracy(especially at lower freqs) but higher latency and vice versa.

    public float binThreshold = 0.005f; //exclude the freq bins with magnitude below this value.

    private float[] previousSpectrumData;
    private float[] frequencies;

    public float lowPassCutoff = 200f;
    public float bandPassCenter = 1000f;
    public bool filterOn = true;

    private double[] lowPassZi = new double[2];

    private static int fs;
    private static int bufferSize;
    int bufferNr;

    private float[] circularBufferLow;

    private int writeIndexLow = 0;

    private readonly object bufferLock = new object(); 
    private static int windowSize;
    bool started = false;
    private Task<SpectralAnalysisResult> currentAnalysisTask = null;
    public int filterSize = 1000;

    MovingAverageFilter spectralCentroidFilter;

    public TMP_Text textLowFreq;

    void Start()
    {
        QualitySettings.vSyncCount = 0; //Disable v-sync so target framerate can be accurate
        Application.targetFrameRate = 60; //clamp the framerate at 60hz

    
        audioSource = GetComponent<AudioSource>();
        AudioSettings.GetDSPBufferSize(out bufferSize, out bufferNr);

        windowSize = bufferSize * (int)System.Math.Pow(2, analysWindowLengthPow);
        circularBufferLow = new float[windowSize]; 

        previousSpectrumData = new float[windowSize];

        // Calculate the frequencies corresponding to each bin

        int numBins = windowSize / 2;
        frequencies = new float[numBins];

        for (int i = 0; i < numBins; i++)
        {
            frequencies[i] = i * AudioSettings.outputSampleRate / (float)windowSize;
        }

        lowPassZi[0] = 0;
        lowPassZi[1] = 0;

        fs = AudioSettings.outputSampleRate;
        spectralCentroidFilter = new MovingAverageFilter(filterSize);
        started = true;
    }

    void FixedUpdate()
    {
        // Only start a new analysis if the previous one is finished.
        if (currentAnalysisTask == null || currentAnalysisTask.IsCompleted)
        {

            float[] analysisBuffer = new float[circularBufferLow.Length];
            lock (bufferLock)
            {
                int idx = writeIndexLow;
                for (int i = 0; i < circularBufferLow.Length; i++)
                {
                    analysisBuffer[i] = circularBufferLow[(idx + i) % circularBufferLow.Length];
                }
            }
            currentAnalysisTask = CalculateSpectralAnalysisAsync(analysisBuffer, "lowFreq");
        }
        if (currentAnalysisTask != null)
        {
            SpectralAnalysisResult result = currentAnalysisTask.Result;

            string analysisParameters = $"{result.Label}\n" +
                                        $"Centroid: {result.SpectralCentroid},\n" +
                                        $"Spread: {result.SpectralSpread},\n" +
                                        $"Entropy: {result.SpectralEntropy},\n" +
                                        $"Spectral Flux: {result.SpectralFlux},\n" +
                                        $"Spectral Tilt: {result.SpectralTilt},\n" +
                                        $"RMS Energy: {result.RMSEnergy},\n" +
                                        $"Zero Crossing Rate: {result.ZeroCrossingRate},\n";

            if (result.Label == "lowFreq")
            {
                textLowFreq.text = analysisParameters;
            }

            currentAnalysisTask = null;
        }
    }
    void OnAudioFilterRead(float[] data, int channels)
    {
        if (started)
        {
            lock (bufferLock)
            {
                for (int i = 0; i < data.Length; i += channels)
                {
                    // Average stereo to mono - this could be extended to support stereo
                    float sample = data[i];
                    if (channels > 1)
                    {
                        sample = (data[i] + data[i + 1]) * 0.5f;
                    }

                    // filtering here - change the filter type or add more filter outputs as desired
                    double lowPassSignal = StateVariableFilter(sample, lowPassCutoff, 1, "LP", lowPassZi);

                    circularBufferLow[writeIndexLow] = (float)lowPassSignal;
                    writeIndexLow = (writeIndexLow + 1) % circularBufferLow.Length;
                }
            }
        }
    }

    // Compute fft and return normalized bins
    void PerformFFT(float[] buffer, float[] spectrum)
    {
        int n = buffer.Length;
        // Determine required length for packed real FFT:
        int dataLength = (n % 2 == 0) ? n + 2 : n + 1;
        float[] data = new float[dataLength];

        Array.Copy(buffer, data, n);
        for (int i = n; i < dataLength; i++)
        {
            data[i] = 0f;
        }

        Fourier.ForwardReal(data, n, FourierOptions.Matlab);

        // Select the gain based on the selected window.
        float coherentGain = 1f;
        switch (fftWindow)
        {
            case FFTWindow.Rectangular:
                coherentGain = 1f;
                break;
            case FFTWindow.Triangle:
                coherentGain = 0.5f;
                break;
            case FFTWindow.Hamming:
                coherentGain = 0.54f;
                break;
            case FFTWindow.Hanning:
                coherentGain = 0.5f;
                break;
            case FFTWindow.Blackman:
                coherentGain = 0.42f;
                break;
            case FFTWindow.BlackmanHarris:
                coherentGain = 0.35875f;
                break;
        }
        int numBins = (n % 2 == 0) ? (n / 2 + 1) : ((n + 1) / 2);

        // Normalize and extract the magnitude spectrum:
        // DC component:
        spectrum[0] = Math.Abs(data[0]) / (n * coherentGain);

        if (n % 2 == 0)
        {
            // For even n, Nyquist bin is stored at data[1].
            spectrum[numBins - 1] = Math.Abs(data[1]) / (n * coherentGain);
            // For bins 1 to (n/2 - 1), use both real and imaginary parts.
            for (int k = 1; k < n / 2; k++)
            {
                float re = data[2 * k];
                float im = data[2 * k + 1];
                float mag = (float)Math.Sqrt(re * re + im * im);
                spectrum[k] = (2 * mag) / (n * coherentGain);
            }
        }
        else
        {
            // For odd n, process bins 1 to numBins - 1.
            for (int k = 1; k < numBins; k++)
            {
                float re = data[2 * k];
                float im = data[2 * k + 1];
                float mag = (float)Math.Sqrt(re * re + im * im);
                spectrum[k] = (2 * mag) / (n * coherentGain);
            }
        }
    }

    private Task<SpectralAnalysisResult> CalculateSpectralAnalysisAsync(float[] samples, string label)
    {
        return Task.Run(() =>
        {
            try
            {
                int n = samples.Length;
                int numBins = (n % 2 == 0) ? (n / 2 + 1) : ((n + 1) / 2);
                float[] spectrum = new float[numBins];

                PerformFFT(samples, spectrum);
                float spectralCentroid = spectralCentroidFilter.AddValue(CalculateSpectralCentroid(spectrum, frequencies));
                float spectralSpread = CalculateSpectralSpread(spectrum, frequencies, spectralCentroid);
                float spectralEntropy = CalculateSpectralEntropy(spectrum);
                float rmsEnergy = CalculateRMSEnergy(samples);
                float zeroCrossingRate = CalculateZeroCrossingRate(samples, fs);
                float spectralFlux = CalculateSpectralFlux(spectrum, previousSpectrumData);
                float spectralTilt = CalculateSpectralTilt(spectrum, frequencies, spectralCentroid, spectralSpread);

                // Update previous spectrum for next frame
                Array.Copy(spectrum, previousSpectrumData, spectrum.Length);

                SpectralAnalysisResult result = new SpectralAnalysisResult()
                {
                    Label = label,
                    SpectralCentroid = spectralCentroid,
                    SpectralSpread = spectralSpread,
                    SpectralEntropy = spectralEntropy,
                    RMSEnergy = rmsEnergy,
                    ZeroCrossingRate = zeroCrossingRate,
                    SpectralFlux = spectralFlux,
                    SpectralTilt = spectralTilt,
                    Spectrum = spectrum
                };

                //Debug.Log("FFT task completed. ZCR is: " + result.ZeroCrossingRate);
                return result;
            }
            catch (Exception ex)
            {
                Debug.LogError("Exception in FFT task: " + ex);
                throw;
            }
        });
    }

    double StateVariableFilter(double input, double fc, double q, string filterType, double[] zi)
    {
        if (filterOn)
        {
            double alpha1 = 2 * Mathf.Sin((float)(Mathf.PI * fc / fs));
            double qInv = 1.0 / q;

            double hp = input - zi[1] - qInv * zi[0];
            double bp = alpha1 * hp + zi[0];
            double lp = alpha1 * bp + zi[1];

            // Update states
            zi[0] = bp;
            zi[1] = lp;

            return filterType switch
            {
                "HP" => hp,
                "BP" => bp,
                "LP" => lp,
                _ => lp,
            };
        }
        else
            return input;
    }

    float CalculateSpectralCentroid(float[] spectrum, float[] freqs, int b1 = 0, int b2 = -1)
    {
        if (b2 == -1) b2 = spectrum.Length / 2 - 1; // Use only the first half of the FFT if it's mirrored - done during prototyping

        float numerator = 0f;
        float denominator = 0f;

        for (int k = b1; k <= b2; k++)
        {
            if (spectrum[k] > binThreshold)
            {
                numerator += frequencies[k] * spectrum[k];
                denominator += spectrum[k];
            }
        }

        return denominator == 0 ? 0 : numerator / denominator;
    }

    float CalculateSpectralSpread(float[] spectrum, float[] freqs, float centroid, int b1 = 0, int b2 = -1)
    {
        if (b2 == -1) b2 = spectrum.Length / 2 - 1;

        float numerator = 0f;
        float denominator = 0f;

        for (int k = b1; k <= b2; k++)
        {
            if (spectrum[k] > binThreshold)
            {
                numerator += Mathf.Pow(freqs[k] - centroid, 2) * spectrum[k];
                denominator += spectrum[k];
            }
        }

        return Mathf.Sqrt(numerator / denominator);
    }

    float CalculateSpectralEntropy(float[] spectrum, int b1 = 0, int b2 = -1)
    {
        if (b2 == -1) b2 = spectrum.Length / 2 - 1;

        float entropy = 0f;
        float sumSpectralValues = 0f;

        // Calculate sum of spectral values
        for (int k = b1; k <= b2; k++)
        {
            if (spectrum[k] > binThreshold)
            {
                sumSpectralValues += spectrum[k];
            }
        }

        // Calculate entropy
        for (int k = b1; k <= b2; k++)
        {
            float p_k = spectrum[k] / sumSpectralValues;
            if (p_k > 0) // Avoid log(0)
            {
                if (spectrum[k] > binThreshold)
                {
                    entropy -= p_k * Mathf.Log(p_k);
                }
            }
        }

        // Normalize entropy
        float normalizationFactor = Mathf.Log(b2 - b1 + 1);
        entropy /= normalizationFactor;

        return entropy;
    }

    float CalculateRMSEnergy(float[] samples)
    {
        float sum = 0f;
        for (int i = 0; i < samples.Length; i++)
        {
            sum += samples[i] * samples[i];
        }
        return Mathf.Sqrt(sum / samples.Length);
    }

    float CalculateZeroCrossingRate(float[] samples, int sampleRate)
    {
        int zeroCrossings = 0;
        for (int i = 1; i < samples.Length; i++)
        {
            if (samples[i - 1] * samples[i] < 0)
            {
                zeroCrossings++;
            }
        }

        float durationInSeconds = (float)samples.Length / sampleRate;
        float zcrHz = zeroCrossings / durationInSeconds / 2.0f;

        return zcrHz;
    }

    float CalculateSpectralFlux(float[] spectrum, float[] previousSpectrum)
    {
        float flux = 0f;
        int halfLength = spectrum.Length / 2;

        for (int i = 0; i < halfLength; i++)
        {
            if (spectrum[i] > binThreshold)
            {
                float value = spectrum[i] - previousSpectrum[i];
                flux += value > 0 ? value : 0;
            }
        }

        return flux;
    }

    float CalculateSpectralTilt(float[] spectrum, float[] frequencies, float spectralCentroid, float spectralSpread, int b1 = 0, int b2 = -1)
    {
        if (b2 == -1) b2 = spectrum.Length / 2 - 1; 

        float numerator = 0f;
        float denominator = 0f;

        for (int k = b1; k <= b2; k++)
        {
            numerator += Mathf.Pow(frequencies[k] - spectralCentroid, 3) * spectrum[k];
            denominator += spectrum[k];
        }

        float skewness = numerator / (Mathf.Pow(spectralSpread, 3) * denominator);

        return skewness;
    }

}