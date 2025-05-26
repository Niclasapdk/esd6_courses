fs = 44100;        % Sampling frequency
fc = 5000;         % Cutoff frequency
Wn = fc/(fs/2);    % Normalize cutoff frequency (Nyquist frequency)

order = 2;         % Filter order (adjustable)

[b, a] = butter(order, Wn, 'low'); % Low-pass IIR filter

freqz([b,a],2^12,fs); % Visualize magnitude and phase
