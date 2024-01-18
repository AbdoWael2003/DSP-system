import math
import cmath
import random
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
import re
import sys
import os

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

import helper_functions as H


WINDOW_WIDTH = 1920
WINDOW_HEIGHT = 1080

mainWindow = tk.Tk()
readyInputsWindow = tk.Tk(); readyInputsWindow.withdraw()
SignalOperationsWindow = tk.Tk(); SignalOperationsWindow.withdraw()
quantizationWindow = tk.Tk(); quantizationWindow.withdraw()
FFTWindow = tk.Tk(); FFTWindow.withdraw()
randomSignalWindow = tk.Tk(); randomSignalWindow.withdraw()
removeDCWindow = tk.Tk(); removeDCWindow.withdraw()
DCTWindow = tk.Tk(); DCTWindow.withdraw()
smoothingWindow = tk.Tk(); smoothingWindow.withdraw()
sharpeningWindow = tk.Tk(); sharpeningWindow.withdraw()
convolutionWindow = tk.Tk(); convolutionWindow.withdraw()
correlationbWindow = tk.Tk(); correlationbWindow.withdraw()
classificationWindow = tk.Tk(); classificationWindow.withdraw()
fileManipulationWindow = tk.Tk(); fileManipulationWindow.withdraw()
resamplingWindow = tk.Tk(); resamplingWindow.withdraw()
ECGClassificationWindow = tk.Tk(); ECGClassificationWindow.withdraw()

FIRWindow = tk.Tk(); FIRWindow.withdraw()



            
        

class FILE:
    def __init__(self):
        pass
    
    def FromPolarToCartesian(X = [], Y = [], export_file = 1):
        if(len(X) == len(Y) == 0):
            Y,X = ReadFile(num_of_columns = 2,meta_data_bits = 1)
        
        real = []
        img = []
        
        for i in range(0, min(len(Y[0]), len(Y[1]))):
            x = cmath.rect(Y[0][i],Y[1][i])
            real.append(x.real); img.append(x.imag)
        
        
        if(export_file):
            path = 'signals/CartesianSignal' + str(len(os.listdir(os.getcwd() + '/signals')) + 1) + '.txt'
            WriteFile(real,img,path,X[0],X[1],X[2])
        return real,img
    
    def FromCartesianToPolar(X = [],Y = [], export_file = 1):
        if(len(X) == len(Y) == 0):
            Y,X = ReadFile(num_of_columns = 2,meta_data_bits = 1)
            
        amps = []
        phases = []
        
        for i in range(0, min(len(Y[0]), len(Y[1]))):
            x = cmath.polar(complex(Y[0][i],Y[1][i]))
            amps.append(x[0]); phases.append(x[1])
            
        if(export_file):
            path = 'signals/PolarSignal' + str(len(os.listdir(os.getcwd() + '/signals')) + 1) + '.txt'
            WriteFile(amps,phases,path,X[0],X[1],X[2])
        return amps,phases
    
    def SignalsPadding(signals, export_file = 1):
        # print(signals[0][1])
        
        # signals must be list of tuples each tuple represent a file
        # the tuple in form of a pair (two elements)
        # the first element of the tuple is a 2d list representing the signal
        # the second element of the tuple is the meta data bits
        
        # converting signals to maps
        
        try:
            number_of_signals = len(signals)
            list_of_maps = [] # each element represent a map for the corresponding signal
            
            minimum_x_position = signals[0][0][0][0]
            maximum_x_position = minimum_x_position
            
            for i,(signal, meta_data) in enumerate(signals):
                list_of_maps.append({})
                
                for j in range(0,min(len(signal[0]),len(signal[1]))):
                    minimum_x_position = signal[0][j] if signal[0][j] < minimum_x_position else minimum_x_position
                    maximum_x_position = signal[0][j] if signal[0][j] > maximum_x_position else maximum_x_position
                    list_of_maps[i].update({signal[0][j]: signal[1][j]})
                    
            returned_signals = []
            
            for i in range(0,number_of_signals):
                returned_signals.append(([[],[]],signals[i][1]))
                
                for x_value in range(int(minimum_x_position),int(maximum_x_position) + 1):
                    y_value = list_of_maps[i].get(x_value)
                    if y_value == None:
                        y_value = 0
                    returned_signals[i][0][0].append(x_value)
                    returned_signals[i][0][1].append(y_value)
                    
                if len(returned_signals[i][1]) != 0:
                    returned_signals[i][1][2] = len(returned_signals[i][0][0])
                  
            
            if(export_file == 1):
                folder_path = 'signals/padded_signals' + str(len(os.listdir(os.getcwd() + '/signals')) + 1)
                os.mkdir(folder_path)
                for i in range(0,number_of_signals):
                    file_path = folder_path + f"/signal{i + 1}.txt"
                    
                    meta_data_bits = returned_signals[i][1]
                    with_meta_data = 1
                    if(len(returned_signals[i][1]) == 0):
                        with_meta_data = 0
                        meta_data_bits.extend([0]*3)
                        
                    WriteFile(returned_signals[i][0][0], returned_signals[i][0][1],file_path,
                            periodic = meta_data_bits[0],
                            type = meta_data_bits[1],
                            N = meta_data_bits[2],
                            meta_data_bits = with_meta_data)
                    
                messagebox.showinfo(message = f"your padded Signals has been exported successfully \n folder path : {folder_path}")
            return returned_signals
            
                
            
        except:
            error = f"Wrong Format for the signals data structure should be in the format 1d[(2d[], 1d[]), (2d[], 1d[]), ...]"
            # print(error + f"\n take a look at the argument \n\n {signals} \n\n" + "\a")
            messagebox.showerror(message = error + "\n look at the consol to see the details")

class FIR:
    
    def _filter_equation(self,fc,n):
        return 2 * fc * math.sin(n * 2 * np.pi * fc) / (n * 2 * np.pi * fc)
    
    def get_h_of_n(self, n):
        return self.get_hd_of_n(n) * self.get_w_of_n(n)
    
    def get_hd_of_n(self, n):
        if self.filter_type == 'lowpass':
            if n == 0: return 2 * self.fc1
            return self._filter_equation(self.fc1,n)
        elif self.filter_type == 'highpass':
            if n == 0: return 1 - 2 * self.fc1
            return -self._filter_equation(self.fc1,n)
        elif self.filter_type == 'bandpass':
            if n == 0: return 2 * (self.fc2 - self.fc1)
            return self._filter_equation(self.fc2,n) - self._filter_equation(self.fc1,n)
        elif self.filter_type == 'bandstop':
            if n == 0: return 1 - 2 * (self.fc2 - self.fc1)
            return self._filter_equation(self.fc1,n) - self._filter_equation(self.fc2,n)
        
    def get_w_of_n(self, n):
        if n == 0: return 1
        
        if self.window == 'rectangular':
            return 1
        elif self.window == 'hanning':
            return 0.5 + 0.5 * math.cos(2 * np.pi * n / self.N)
        elif self.window == 'hamming':
            return 0.54 + 0.46 * math.cos(2 * np.pi * n / self.N)
        elif self.window == 'blackman':
            return 0.42 + 0.5 * math.cos(2 * np.pi * n / (self.N - 1)) + 0.08 * math.cos(4 * np.pi * n / (self.N - 1))
        
    
    def get_fc(self):
        if self.filter_type == 'lowpass':
            return self.fc1 + self.transition_width / 2
        elif self.filter_type == 'highpass':
            return self.fc1 - self.transition_width / 2
        elif self.filter_type == 'bandpass':
            # pair of(fc1,fc2)
            return(self.fc1 - self.transition_width / 2, self.fc2 + self.transition_width / 2) 
        elif self.filter_type == 'bandstop':
            return(self.fc1 + self.transition_width / 2, self.fc2 - self.transition_width / 2)
        
    def get_window_type(self):
        if self.stopBandAttenuation <= 21:
            self.N = 0.9 / self.transition_width
            self.N = math.ceil(self.N); self.N = self.N + 1 if self.N % 2 == 0 else self.N
            return 'rectangular'
        if self.stopBandAttenuation <= 44:
            self.N = 3.1 / self.transition_width
            self.N = math.ceil(self.N); self.N = self.N + 1 if self.N % 2 == 0 else self.N
            return 'hanning'
        if self.stopBandAttenuation <= 53:
            self.N = 3.3 / self.transition_width
            self.N = math.ceil(self.N); self.N = self.N + 1 if self.N % 2 == 0 else self.N
            return 'hamming'
        if self.stopBandAttenuation <= 74:
            self.N = 5.5 / self.transition_width
            self.N = math.ceil(self.N); self.N = self.N + 1 if self.N % 2 == 0 else self.N
            return 'blackman'
        else:
            messagebox.showerror(message = "Invalid stopband attenuation")
        

    def __init__(self, fc1, fs, stopBandAttenuation,transition_width,fc2 = None, filter_type = 'low pass'):
        self.filter_type = filter_type
        self.fs = fs
        self.transition_width = transition_width / fs
        self.fc1 = fc1 / fs
        if(fc2 != None):
            self.fc2 = fc2 / fs
        
        f = self.get_fc()
        
        two_cut_frequencies = (self.filter_type == 'bandpass' or
                               self.filter_type == 'bandstop')
        
        self.fc1 = f[0] if two_cut_frequencies else f
        
        self.fc2 = f[1] if two_cut_frequencies else None
        
        
        self.N = 0
        
        self.stopBandAttenuation = stopBandAttenuation
        
        self.window = self.get_window_type()
        
        
def Resampling(L, M, X = [], Y = []):
    new_Y = []
    new_X = []
    if M == L == 0:
        messagebox.showerror("Invalid ReSampling parameters L = 0, M = 0 \n")
    if L:
        i = 0
        temp = X[0]
        while(i < len(Y)):
            new_X.append(temp); new_Y.append(Y[i]); temp += 1
            if i != len(Y) - 1:
                for j in range(0,L - 1): 
                    new_X.append(temp)
                    new_Y.append(0)
                    temp += 1
            i += 1
        return new_X, new_Y
    else:
        new_Y = Y; new_X = X
    if M:
        new_Y = new_Y[::M];  new_X = new_X[::M]
    
    return new_X, new_Y

def Conv(X1,Y1,X2,Y2):
            
    mini = (X1[0] + X2[0])
    maxi = (X1[len(X1) - 1] + X2[len(X2) - 1])
    
    Y3 = []
    X3 = []
    
    temp = mini
    while(temp <= maxi):
        X3.append(temp)
        temp = temp + 1
    
    x = {}
    h = {}
    
    for i in range(0,len(Y1)):
        x.update({X1[i]: Y1[i]})
    
    for i in range(0,len(Y2)):
        h.update({X2[i]: Y2[i]})
            
    n = mini
    while(n <= maxi):
        k = int(X1[0])
        y_of_n = 0
        while(n - k >= X2[0] and k <= X1[len(X1) - 1]):
            temp1 = x.get(k)
            temp2 = h.get(n - k)
            
            if(temp1 == None):
                temp1 = 0
            if(temp2 == None):
                temp2 = 0
                
            y_of_n += temp1 * temp2
            
            k = k + 1
        Y3.append(y_of_n)
        n = n + 1
    return X3,Y3

def Corr(Y1,Y2):
        
    r = []
    normalized_r = []
    
    N = len(Y1)
        
        
    for j in range(0,N):
        temp = 0
        for n in range(0,N):
            temp += (Y1[n] * Y2[(n + j) % N])
        r.append(temp / N)
    
    sum_of_x1_squared = 0
    sum_of_x2_squared = 0
    
    for y in Y1: sum_of_x1_squared += y*y
    for y in Y2: sum_of_x2_squared += y*y
    
    denominator = math.sqrt(sum_of_x1_squared * sum_of_x2_squared) / N
    
    for ri in r:
       normalized_r.append(ri / denominator)
       
    return r, normalized_r

def average_signals(signals): # it takes 2d list each element is a list representing a signal
    try:
        averaged_signal = []
        for i in range(0, len(signals[0])):
            a_of_i = 0
            for j in range(0, len(signals)):
                a_of_i += signals[j][i]
            averaged_signal.append(a_of_i)
        return averaged_signal
    except:
        messagebox.showerror(message = "Error in average_signals() function \n note that the signals must be equal in length")
        # for i in range(0,len(signals)):
            # print(f"the length of signal {i + 1} is {len(signals[i])}")
            
        
def OpenOnce(window_name, Draw):
    try:
        if(globals()[window_name].state() == 'normal' or globals()[window_name].state() == 'zoomed'):
            messagebox.showerror(message = f"{window_name} already opened!", title="Error")
            globals()[window_name].state("zoomed")
            return 0
        else:
           globals()[window_name].state("zoomed")
           Draw(globals()[window_name])
           return 1
    except:
        globals()[window_name] = tk.Tk()
        globals()[window_name].geometry("1300x700")
        globals()[window_name].state("zoomed")
        Draw(globals()[window_name])
        return 1
    
def WriteFile(X, Y, path, periodic, type, N, meta_data_bits = 1):
    
    with open(path,'w') as f:
        if meta_data_bits == 1:
            f.write(f"{periodic}\n{type}\n{N}\n");
        for i in range(0, len(X)):
            f.write(f"{X[i]} {Y[i]}\n")
            
    messagebox.showinfo(message = f"File has been exported successfully \n path = {path}")
                
def ReadFile(num_of_columns = 2,meta_data_bits = 1, path = 'none'):
    try:
        if path == 'none':
            filename = filedialog.askopenfilename()
        else:
            filename = path
    except:
        messagebox.showerror(message = "Something wrong with the file name or extension\n may be the file dosn't exist !")
        return None
    if filename == "": return None
    
    # print(filename)
    
    fields = []
    meta_data = []
    
    for i in range(0, num_of_columns):
        fields.append([])
    
    try:
        with open(filename, 'r') as f:
            n = 0
            for i, line in enumerate(f):
                if i < 3 and meta_data_bits:
                    meta_data.append(int(line))
                if i == 2 and meta_data_bits:
                    n = int(line)
                    # print(n)
                if (3 <= i < n + 3) or meta_data_bits == 0:
                    pattern = '[^0-9 ,.eE-]'
                    line = re.sub(pattern,'',line)
                    line.replace(',',' ')
                    line = re.sub(r'\s+',' ',line)
                    # print(line)
                    for j, entry in enumerate(line.split()):
                        fields[j].append(float(entry))
    except:
        messagebox.showerror(message = "Something Wrong with the file format..!\n take a look at the file \n")
        os.startfile(filename)
        
    for i,column in enumerate(fields):
        try:
            fields[i] = list(map(float, column))
        except:
            messagebox.showerror(message = "only float values are allowed inside columns !")
    
    return fields,meta_data # [0] grid[0] first_column  , [1] 3 bits meta_data

def ReadDirectory(num_of_columns = 2,meta_data_bits = 0, path = 'none'):
    path = filedialog.askdirectory()
    folder = [] # an element is tuple of (2d list, meta_data) for a single file
    for file in os.listdir(path):
        file_path = path + f'\{file}'
        fields, meta_data = ReadFile(num_of_columns,meta_data_bits,file_path)
        folder.append(list(zip([fields,],[meta_data,]))[0]) # (fields,meta_data)
    return folder

def display():
    x,y = ReadFile()[0]
    
    plt.plot(x,y)
    plt.show()

def ReadyInputs():
    def ReadData():
        try:
            amp = float(Draw.frame.winfo_children()[6].get())
            af = float(Draw.frame.winfo_children()[7].get())
            fs = float(Draw.frame.winfo_children()[8].get())
            phase = float(Draw.frame.winfo_children()[9].get())
            n = int(Draw.frame.winfo_children()[10].get())
            signal_type = Draw.frame.winfo_children()[11].get()
        except:
            messagebox.showerror(message = "Invalid Data")
        if fs < 2 * af:
            messagebox.showerror(title="Alias",message = "Alias Probelm try biger sampling frequency")
            return
        
        time = np.arange(0.0,1,0.001) # <=====================
        Xa = []
        Xd = []
        
        for i in range(1,n + 1):
            if signal_type == 'sin':
                Xd.append(amp * np.sin(2 * np.pi * af / fs * i + phase))
            else:
                Xd.append(amp * np.cos(2 * np.pi * af / fs * i + phase))
        
        
        for t in time:
            if signal_type == 'sin':
                Xa.append(amp * np.sin(2 * np.pi * af * t + phase)) 
            else:
                Xa.append(amp * np.cos(2 * np.pi * af * t + phase))
            
        fig,ax = plt.subplots(nrows = 2)
        ax[0].plot(time,Xa); ax[1].stem(range(1,n + 1),Xd)
        
        plt.show()
        
    def Draw(x):
        x.deiconify()
        Draw.frame = tk.Frame(x); Draw.frame.pack()
        amplitude = tk.Label(Draw.frame,text = 'Amplitude'); amplitude.grid(row = 0, column = 0)
        analogFrequency = tk.Label(Draw.frame,text = 'Analog Frequency'); analogFrequency.grid(row = 1, column = 0)
        samplingFrequency = tk.Label(Draw.frame,text = 'Sampling Frequency'); samplingFrequency.grid(row = 2, column = 0)
        phaseShift = tk.Label(Draw.frame,text = 'Phase Shift'); phaseShift.grid(row = 3, column = 0)
        numberOfSamples = tk.Label(Draw.frame,text = 'Number of Samples'); numberOfSamples.grid(row = 4, column = 0)
        signalGenerator = tk.Label(Draw.frame,text = 'Signal Generator'); signalGenerator.grid(row = 5, column = 0)

        amplitudeEntry = tk.Entry(Draw.frame); amplitudeEntry.grid(row = 0, column = 1)
        analogFrequencyEntry = tk.Entry(Draw.frame); analogFrequencyEntry.grid(row = 1, column = 1)
        samplingFrequencyEntry = tk.Entry(Draw.frame); samplingFrequencyEntry.grid(row = 2, column = 1)
        phaseShiftEntry = tk.Entry(Draw.frame); phaseShiftEntry.grid(row = 3, column = 1)
        numberOfSamplesEntry = tk.Entry(Draw.frame); numberOfSamplesEntry.grid(row = 4, column = 1)
        signalGeneratorEntry = ttk.Combobox(Draw.frame,values = ['sin','cos']); signalGeneratorEntry.grid(row = 5, column = 1)
        
        displayButton = tk.Button(Draw.frame,width=40,height=4,text = "Display Signal",command = ReadData);displayButton.grid(row=6)
        
        
        
        for widget in Draw.frame.winfo_children():
            widget.grid_configure(padx = 10,pady = 10)
        
        x.mainloop()
    OpenOnce('readyInputsWindow',Draw)

def SignalOperations():
    SignalOperations.currentSignal = 0
    SignalOperations.x = []
    SignalOperations.y = []
    
    def ExportSignal():
        path = ""
        try:
            path = 'signals/OPERATED_SIGNAL' + str(len(os.listdir(os.getcwd() + '/signals')) + 1) + '.txt'
        except:
            messagebox.showerror(message = f"Can't export the signal at this given path: \"{path}\" \n Check if there's missing files or folders in your directory")
        WriteFile(SignalOperations.x,SignalOperations.y,path,0,0,len(SignalOperations.x))
        messagebox.showinfo(message = f"Signal has been exported successfully\n with file path : \"{path}\" ")

    def Add():
        x,y = ReadFile()[0]
        if SignalOperations.currentSignal:
            for i in range(0,min(len(SignalOperations.x),len(x))):
                SignalOperations.y[i] += y[i]
        else:
            SignalOperations.x = x; SignalOperations.y = y
            SignalOperations.currentSignal = 1
            Draw.currentSignal.winfo_children()[0]['text'] = 'Current Signal'
            exportButton =  tk.Button(Draw.frame, width = 30,height = 4,text = "Export Signal",command = ExportSignal); exportButton.grid(row = 0, column = 2,padx = 2,pady = 2)

        
        fig,ax = plt.subplots(nrows = 1,figsize = (9,8))
        ax.plot(SignalOperations.x,SignalOperations.y)
        canvas = FigureCanvasTkAgg(fig, master = Draw.currentSignal)   
        canvas.draw() 
        canvas.get_tk_widget().grid(row = 1, column = 0)
        
    def Subtract():
        x,y = ReadFile()[0]

        if SignalOperations.currentSignal:
            for i in range(0,min(len(SignalOperations.x),len(x))):
                SignalOperations.y[i] -= y[i]
        else:
            SignalOperations.x = x; SignalOperations.y = y
            SignalOperations.currentSignal = 1
            exportButton =  tk.Button(Draw.frame, width = 30,height = 4,text = "Export Signal",command = ExportSignal); exportButton.grid(row = 0, column = 2,padx = 2,pady = 2)

        
        fig,ax = plt.subplots(nrows = 1,figsize = (9,8))
        ax.plot(SignalOperations.x,SignalOperations.y)
        canvas = FigureCanvasTkAgg(fig, master = Draw.currentSignal)   
        canvas.draw() 
        canvas.get_tk_widget().grid(row = 1, column = 0)
        
    def Multiply():
        
        if SignalOperations.currentSignal == 0:
            return
        
        scale = float(Draw.frame.winfo_children()[4].get())
        
        for i in range(0,len(SignalOperations.y)):
            SignalOperations.y[i] *= scale
            
        fig,ax = plt.subplots(nrows = 1,figsize = (9,8))
        ax.plot(SignalOperations.x,SignalOperations.y)
        canvas = FigureCanvasTkAgg(fig, master = Draw.currentSignal)   
        canvas.draw() 
        canvas.get_tk_widget().grid(row = 1, column = 0)
        
    
    def Squaring():
        
        if SignalOperations.currentSignal == 0:
            return
        
        
        for i in range(0,len(SignalOperations.y)):
            SignalOperations.y[i] *= SignalOperations.y[i]
            
        fig,ax = plt.subplots(nrows = 1,figsize = (9,8))
        ax.plot(SignalOperations.x,SignalOperations.y)
        canvas = FigureCanvasTkAgg(fig, master = Draw.currentSignal)   
        canvas.draw() 
        canvas.get_tk_widget().grid(row = 1, column = 0)
        
    def Shift(auto = 0):
        if SignalOperations.currentSignal == 0:
            return 
        unit = auto if auto else int(Draw.frame.winfo_children()[7].get())
    
        
        if SignalOperations.currentSignal == 0 or unit == 0:
            return
        
        if(unit < 0):
            for i in range(len(SignalOperations.y) - 1,0,-1):
                if(i + unit >= 0):
                    SignalOperations.y[i] = SignalOperations.y[i + unit]
                else:
                    SignalOperations.y[i] = 0
        else:
             for i in range(0,len(SignalOperations.y)):
                if(i + unit < len(SignalOperations.y)):
                    SignalOperations.y[i] = SignalOperations.y[i + unit]
                else:
                    SignalOperations.y[i] = 0
            
        fig,ax = plt.subplots(nrows = 1,figsize = (9,8))
        ax.plot(SignalOperations.x,SignalOperations.y)
        canvas = FigureCanvasTkAgg(fig, master = Draw.currentSignal)   
        canvas.draw() 
        canvas.get_tk_widget().grid(row = 1, column = 0)
        
    def normalize_list(list1, lower_bound, upper_bound):
        if SignalOperations.currentSignal == 0:
            return 
        # Find the minimum and maximum values in the list.
        min_value = min(list1)
        max_value = max(list1)

        # Calculate the range of the list.
        list_range = max_value - min_value

        # Calculate the desired range.
        desired_range = upper_bound - lower_bound

        # Calculate the normalization factor.
        normalization_factor = desired_range / list_range

        # Normalize the values in the list.
        normalized_list = []
        for value in list1:
            normalized_value = (value - min_value) * normalization_factor + lower_bound
            normalized_list.append(normalized_value)

        return normalized_list
    def Normalize():
            
        if SignalOperations.currentSignal == 0:
            return
        try:
            mini = float(Draw.frame.winfo_children()[10].get())
            maxi = float(Draw.frame.winfo_children()[12].get())
        except:
            messagebox.showerror(title = "Invalid Range",message = "Please specify the starting and ending ranges correctly\n")

        SignalOperations.y = normalize_list(SignalOperations.y,mini,maxi)
            
        fig,ax = plt.subplots(nrows = 1,figsize = (9,8))
        ax.plot(SignalOperations.x,SignalOperations.y)
        canvas = FigureCanvasTkAgg(fig, master = Draw.currentSignal)   
        canvas.draw() 
        canvas.get_tk_widget().grid(row = 1, column = 0)
        
    def Accumulation():
        if SignalOperations.currentSignal == 0:
            return 
        for i in range(1,len(SignalOperations.y)):
            SignalOperations.y[i] += SignalOperations.y[i - 1]
            
        fig,ax = plt.subplots(nrows = 1,figsize = (9,8))
        ax.plot(SignalOperations.x,SignalOperations.y)
        canvas = FigureCanvasTkAgg(fig, master = Draw.currentSignal)   
        canvas.draw() 
        canvas.get_tk_widget().grid(row = 1, column = 0)
        
    def Folding():
        if SignalOperations.currentSignal == 0:
           return

        for i in range(0,len(SignalOperations.x)):
            SignalOperations.x[i] *= -1
        
        pairs = zip(SignalOperations.x,SignalOperations.y)
        
        pairs = sorted(pairs)
        
        i = 0
        for x,y in pairs:
            SignalOperations.x[i] = x
            SignalOperations.y[i] = y
            i = i + 1
            
        fig,ax = plt.subplots(nrows = 1,figsize = (9,8))
        ax.plot(SignalOperations.x,SignalOperations.y)
        canvas = FigureCanvasTkAgg(fig, master = Draw.currentSignal)   
        canvas.draw() 
        canvas.get_tk_widget().grid(row = 1, column = 0)
        
    def Advance():
        if SignalOperations.currentSignal == 0:
            return
        try:
            unit = int(Draw.frame.winfo_children()[19].get())
        except:
            messagebox.showerror(message = "Invalid Data")
        if(unit < 0):
            messagebox.showerror(message = "Please specify the absolute value only")
            return
        Shift(unit)

    def Delay():
        if SignalOperations.currentSignal == 0:
            return
        try:
            unit = int(Draw.frame.winfo_children()[19].get())
        except:
            messagebox.showerror(message = "Invalid Data")
        if(unit < 0):
            messagebox.showerror(message = "Please specify the absolute value only")
            return
        Shift(-1 * unit)

    
    def MultiplySignal():
        if SignalOperations.currentSignal == 0:
            return
        Y = ReadFile(num_of_columns = 2,meta_data_bits = 1)[0][1]
        
        for i in range(0,min(len(SignalOperations.y),len(Y))):
            SignalOperations.y[i] *= Y[i]
            
        
        fig,ax = plt.subplots(nrows = 1,figsize = (9,8))
        ax.plot(SignalOperations.x,SignalOperations.y)
        canvas = FigureCanvasTkAgg(fig, master = Draw.currentSignal)   
        canvas.draw() 
        canvas.get_tk_widget().grid(row = 1, column = 0)
        
            
    def Conjugate():
        if SignalOperations.currentSignal == 0:
            return
        
        SignalOperations.y = list(map(lambda x: x * -1, SignalOperations.y))
        
        fig,ax = plt.subplots(nrows = 1,figsize = (9,8))
        ax.plot(SignalOperations.x,SignalOperations.y)
        canvas = FigureCanvasTkAgg(fig, master = Draw.currentSignal)   
        canvas.draw() 
        canvas.get_tk_widget().grid(row = 1, column = 0)
    
    
    def ComplexMultiplication():
        if SignalOperations.currentSignal == 0:
            return
        signal2 = ReadFile(num_of_columns = 2,meta_data_bits = 1)[0]
        
        for i in range(0,min(min(len(SignalOperations.y), len(SignalOperations.x)),min(len(signal2[0]), len(signal2[1])))):
            SignalOperations.x[i] *= signal2[0][i]; SignalOperations.y[i] *= signal2[1][i]
        
        fig,ax = plt.subplots(nrows = 1,figsize = (9,8))
        ax.plot(SignalOperations.x,SignalOperations.y)
        canvas = FigureCanvasTkAgg(fig, master = Draw.currentSignal)   
        canvas.draw() 
        canvas.get_tk_widget().grid(row = 1, column = 0)
    
    def Draw(x):
        x.deiconify()
        
        Draw.mainFrame = tk.Frame(x)
        
        Draw.frame = tk.Frame(x); Draw.frame.grid(sticky = "news", row = 0,column = 0)
        Draw.currentSignal = tk.Frame(x); Draw.currentSignal.grid(row = 0,column = 1,padx= 50)
        
        add = tk.Button(Draw.frame,width=30,height = 4,text = "Add",command = Add); add.grid(row = 0, column = 0,pady = 2)
        subtract = tk.Button(Draw.frame,width=30,height = 4,text = "Subtract",command = Subtract); subtract.grid(row = 1, column = 0,pady = 2)
        multiply =  tk.Button(Draw.frame,width = 30,height = 4,text = "Multiply",command = Multiply);multiply.grid(row = 2, column = 0,pady = 2)
        multiplyALabel = tk.Label(Draw.frame,font = ("Arial", 15),text = "by"); multiplyALabel.grid(row = 2, column = 1)
        multiplyEntry = tk.Entry(Draw.frame); multiplyEntry.grid(row = 2,column = 1)
        squaring =  tk.Button(Draw.frame,width = 30,height = 4,text = "Squaring",command = Squaring);squaring.grid(row = 3, column = 0,pady = 2)
        shift_by_2 =  tk.Button(Draw.frame,width = 30,height = 4,text = "Shift",command = Shift);shift_by_2.grid(row=4, column = 0,pady = 2)
        shift_by_2Entry = tk.Entry(Draw.frame); shift_by_2Entry.grid(row = 5,column = 0)

        normalize =  tk.Button(Draw.frame,width = 30,height = 4,text = "Normalize",command = Normalize);normalize.grid(row=6, column = 0,pady = 2)
        
        normalizeALabel = tk.Label(Draw.frame,font = ("Arial", 15),text = "Starting Range"); normalizeALabel.grid(row = 5, column = 1)
        normalizeAEntry = tk.Entry(Draw.frame); normalizeAEntry.grid(row = 6,column = 1)
        normalizeBLabel = tk.Label(Draw.frame,font = ("Arial", 15),text = "Ending Range"); normalizeBLabel.grid(row = 5, column = 2)
        normalizeBEntry = tk.Entry(Draw.frame); normalizeBEntry.grid(row = 6,column = 2)
        
        accumulation =  tk.Button(Draw.frame,width = 30,height = 4,text = "Accumulation",command=Accumulation);accumulation.grid(row = 7, column = 0,pady = 2)
        
        signalLabel = tk.Label(Draw.currentSignal,font = ("Arial", 25),text = ""); signalLabel.grid(row = 0, column = 0)
        
        foldingButton =  tk.Button(Draw.frame, width = 30,height = 4,text = "Folding",command = Folding); foldingButton.grid(row = 8, column = 0,pady = 2)
        advanceButton =  tk.Button(Draw.frame, width = 30,height = 4,text = "Advance",command = Advance); advanceButton.grid(row = 9, column = 0,pady = 2)
        delayButton =  tk.Button(Draw.frame, width = 30,height = 4,text = "Delay",command = Delay); delayButton.grid(row = 10, column = 0,pady = 2)

        advanceLabel = tk.Label(Draw.frame, font = ("Arial", 15),text = "by"); advanceLabel.grid(row = 9, column = 1)
        delayLabel = tk.Label(Draw.frame, font = ("Arial", 15),text = "by"); delayLabel.grid(row = 10, column = 1)

        advanceBEntry = tk.Entry(Draw.frame); advanceBEntry.grid(row = 9,column = 2)
        delayEntry = tk.Entry(Draw.frame); delayEntry.grid(row = 10,column = 2)
        
        multiplySignalButton =  tk.Button(Draw.frame, width = 30,height = 4,text = "Multiply Signal",command = MultiplySignal); multiplySignalButton.grid(row = 0, column = 1,pady = 2)

        complexSignalMultiplicationButton =  tk.Button(Draw.frame, width = 30,height = 4,text = "Complex Signal Multiplication",command = ComplexMultiplication); complexSignalMultiplicationButton.grid(row = 1, column = 1,pady = 2)
        conjugateButton =  tk.Button(Draw.frame, width = 30,height = 4,text = "Conjugate",command = Conjugate); conjugateButton.grid(row = 1, column = 2,pady = 2)

        
        # for widget in Draw.frame.winfo_children():
        #     widget.grid_configure(padx = 10,pady = 10)
            
        # for widget in Draw.currentSignal.winfo_children():
        #     widget.grid_configure(padx = 10,pady = 10)
        
        x.mainloop()
    
    OpenOnce('SignalOperationsWindow', Draw)

def Quantization():
    
    y_value = [-1.22, 1.5, 3.24, 3.94, 2.20, -1.10, -2.26, -1.88, -1.2]
    
    maximum = y_value[0]
    minimum = y_value[0]
    
    for i in y_value:
        maximum = max(maximum,i)
        minimum = min(minimum,i)
        
    def ReadData():
        try:
            mode = Draw.frame.winfo_children()[1].get()
            value = int(Draw.frame.winfo_children()[3].get())
        except:
            messagebox.showerror(message = "Invalid Data")
        # print(mode)
        # print(value)
        
        levels = 0
        
        if mode == "numbers":
            levels = value
        else:
            levels = pow(2,value)
        
        delta = (maximum - minimum) / levels
        
        ranges = []
        
        last_z = minimum 
        for i in range(0,levels):
            ranges.append([round(last_z,2),round(last_z + delta,2)])
            last_z = last_z + delta
            
        midPoints = []
        for start,end in ranges:
           midPoints.append((start + end) / 2)
        
        new_y = []
        interval_index = []
        errors = []
        
        averagePowerError = 0
        for val in y_value:
            for i in range(0,levels):
                if  ranges[i][0] <= val <= ranges[i][1]:
                    if new_y == ranges[i][0] and i != 0:
                        new_y.append(midPoints[i - 1])
                    else:
                        new_y.append(midPoints[i])
                    interval_index.append(i)
                    errors.append(midPoints[i] - val)
                    averagePowerError += pow((midPoints[i] - val), 2)
                    break
                
        with open("quantization_table.txt",'w') as file:
            file.write(f"n\t\t\tx(n)\t\t\tinterval index\t\t\txq(n)\t\t\teq(n) = xq(n) - x(n)\t\t\teq^2\n")
            for i in range(0,9):
                file.write(f"{i}\t\t\t{round(y_value[i],3)}\t\t\t{interval_index[i]}\t\t\t           {round(new_y[i],3)}\t\t\t{round(errors[i],3)}\t\t\t{round(pow(errors[i],2),3)}\n")
        
        errorLabel = tk.Label(Draw.frame,text = f'Mean Square Error\n{round(averagePowerError / 9,6)}',font = ("Arial", 25) ); errorLabel.grid(row = 2, column = 1)

        fig,ax = plt.subplots(nrows = 2)
        ax[0].stem(y_value);ax[0].set_ylabel("Unquantized"); ax[0].grid(axis = 'y')
        ax[1].stem(new_y);ax[1].set_ylabel("Quantized"); ax[1].grid(axis = 'y')

        canvas = FigureCanvasTkAgg(fig, master = Draw.currentSignal)   
        canvas.draw() 
        canvas.get_tk_widget().grid(row = 1, column = 0)
        
       
    def Draw(x):
        
        
        x.deiconify()
        Draw.frame = tk.Frame(x); Draw.frame.grid(row = 0)
        Draw.currentSignal = tk.Frame(x); Draw.currentSignal.grid(row = 1)
        
        fig,ax = plt.subplots(nrows = 1)
        ax.stem(y_value)
        ax.grid(axis = 'y')
        canvas = FigureCanvasTkAgg(fig, master = Draw.currentSignal)   
        canvas.draw() 
        canvas.get_tk_widget().grid(row = 1, column = 0)

        
        modeLabel = tk.Label(Draw.frame,text = 'Mode'); modeLabel.grid(row = 0, column = 0)
        mode = ttk.Combobox(Draw.frame,values = ['numbers','bits']); mode.grid(row = 0, column = 1)
        
        valueLabel = tk.Label(Draw.frame,text = 'Value'); valueLabel.grid(row = 1, column = 0)
        valueEntry = tk.Entry(Draw.frame); valueEntry.grid(row = 1, column = 1)
        
        
        displayButton =  tk.Button(Draw.frame,width = 30,height = 4,text = "Display",command = ReadData); displayButton.grid(row = 2)
  
        for widget in Draw.frame.winfo_children():
            widget.grid_configure(padx = 10,pady = 10)
        
        x.mainloop()
    OpenOnce('quantizationWindow',Draw)

def FFT():
    
    FFT.is_harmonic = 0
    FFT.signal_is_present = 0
    FFT.mode = 'none'
    FFT.fs = 0
    FFT.X = []
    FFT.Y = []
    FFT.N = 0
    
    FFT.f = []
    FFT.amps = []
    FFT.phases = []
    FFT.fig,FFT.ax = plt.subplots(nrows = 3)
    FFT.idftResult = []
    
    
    def ExportSignal():
        path = ""
        try:
            path = 'signals/FFT' + str(len(os.listdir(os.getcwd() + '/signals')) + 1) + '.txt'
        except:
            messagebox.showerror(message = f"Can't export the signal at this given path: \"{path}\" \n Check if there's missing files or folders in your directory")
        if(FFT.mode == "DFT"):
            WriteFile(FFT.amps,FFT.phases,path,0,0,len(FFT.amps))
        elif(FFT.mode == 'IDFT'):
            WriteFile(list(range(0,len(FFT.Y))),FFT.Y,path,0,0,len(FFT.Y))
        else:
            messagebox.showerror(message = f"Invalid Signal")
            return
        messagebox.showinfo(message = f"Signal has been exported successfully\n with file path : \"{path}\" ")

    
    def ReadData():
        try:
            FFT.fs = float(Draw.options.winfo_children()[1].get())
            FFT.mode = Draw.options.winfo_children()[3].get() 
            FFT.X,FFT.Y = ReadFile()[0]
        except:
            messagebox.showerror(message = "Invalid Data")
        if FFT.fs == 0:
            messagebox.showerror(title = "Error",message = "divid by zero")
            return
        if FFT.mode == "":
            messagebox.showerror(title = "Error",message = "Invalid mode")
            return
        
        
        FFT.N = len(FFT.X)
        
        FFT.fig,FFT.ax = plt.subplots(nrows = 3)
        for widget in Draw.signalsPlots.winfo_children():
            widget.destroy()
            
      
        
        
        Draw.options.winfo_children()[5]['text'] = "apply " + FFT.mode

        
        if FFT.mode == 'DFT':
            FFT.signal_is_present = 1
            FFT.is_harmonic = 0
            FFT.ax[0].stem(FFT.X,FFT.Y); FFT.ax[0].set_ylabel("x[n]")
            Draw.modifySignals.winfo_children()[1]['values'] = []
            canvas = FigureCanvasTkAgg(FFT.fig, master = Draw.signalsPlots)   
            canvas.draw() 
            canvas.get_tk_widget().pack()
            
        else:
            FFT.is_harmonic = 1
            
            FFT.amps.clear(); FFT.phases.clear(); FFT.f.clear()
            
            last_f = 0
            for i in range(0,FFT.N):
                if FFT.X[i]:
                    FFT.amps.append(FFT.X[i]); FFT.phases.append(FFT.Y[i]); FFT.f.append(last_f + 2 * np.pi * FFT.fs / FFT.N); last_f = FFT.f[len(FFT.f) - 1]
                                      
            FFT.ax[1].bar(FFT.f,FFT.amps); FFT.ax[1].set_ylabel("Amplitude")
            FFT.ax[2].bar(FFT.f,FFT.phases); FFT.ax[2].set_ylabel("Phase Shift")
            
            canvas = FigureCanvasTkAgg(FFT.fig, master = Draw.signalsPlots)   
            canvas.draw() 
            canvas.get_tk_widget().pack()
            
            Draw.modifySignals.winfo_children()[1]['values'] = list(range(0,len(FFT.amps)))
    
    def ApplyTransformation():
      
        if len(Draw.signalsPlots.winfo_children()) == 0:
            messagebox.showerror(title = "Error",message = "There is no Signal to Transform !")
            return
        
        FFT.fig,FFT.ax = plt.subplots(nrows = 3)
        for widget in Draw.signalsPlots.winfo_children():
            widget.destroy()
        
        FFT.amps.clear(); FFT.phases.clear(); FFT.f.clear(); FFT.idftResult.clear()
        
        if FFT.mode == "DFT":
            FFT.is_harmonic = 1
            
            for k in range(0,FFT.N):
                x_of_k = complex(0,0)
                
                for n in range(0,FFT.N):
                    theta = 2 * np.pi * k * n / FFT.N
                    temp = complex(math.cos(theta), -1 * math.sin(theta))
                    temp *= FFT.Y[n]
                    
                    x_of_k += temp
                
                A, phase = cmath.polar(x_of_k)
                if A:
                    FFT.amps.append(A)
                    FFT.phases.append(phase)
            
            last_f = 0
            for i in range(0,len(FFT.amps)):
                FFT.f.append(last_f + 2 * np.pi * FFT.fs / FFT.N); last_f = FFT.f[len(FFT.f) - 1]
                    
            Draw.modifySignals.winfo_children()[1]['values'] = list(range(0,len(FFT.amps)))
            
            # print(FFT.Y)
            # print(FFT.amps)
            # print(FFT.phases)
            # print(FFT.f)
            
            FFT.ax[0].stem(FFT.Y); FFT.ax[0].set_ylabel("x[n]")
            FFT.ax[1].bar(FFT.f,FFT.amps); FFT.ax[1].set_ylabel("Amplitude")
            FFT.ax[2].bar(FFT.f,FFT.phases); FFT.ax[2].set_ylabel("Phase Shift")
            
            canvas = FigureCanvasTkAgg(FFT.fig, master = Draw.signalsPlots)   
            canvas.draw() 
            canvas.get_tk_widget().pack()
            path = 'signals/DFT_OUTPUT' + str(len(os.listdir(os.getcwd() + '/signals')) + 1) + '.txt'
            WriteFile(FFT.amps,FFT.phases,path,0,1,len(FFT.amps))
            
        else:
            FFT.signal_is_present = 1
            for n in range(0,FFT.N):
                x_of_n = complex(0,0)
                
                for k in range(0,FFT.N):
                    theta = 2 * np.pi * k * n / FFT.N
                    temp = complex(math.cos(theta), math.sin(theta))
                    temp *= cmath.rect(FFT.X[k], FFT.Y[k])
            
                    x_of_n += temp
                    
                x_of_n /= FFT.N
                FFT.idftResult.append(x_of_n)
            
            last_f = 0
            for i in range(0,FFT.N):
                if FFT.X[i]:
                    FFT.amps.append(FFT.X[i]); FFT.phases.append(FFT.Y[i]); FFT.f.append(last_f + 2 * np.pi * FFT.fs / FFT.N); last_f = FFT.f[len(FFT.f) - 1]
                             
            # print(FFT.Y)
            # print(FFT.amps)
            # print(FFT.phases)
            # print(FFT.f)
            # print(FFT.idftResult)
            
            FFT.ax[0].stem(FFT.idftResult); FFT.ax[0].set_ylabel("x[n]")
            FFT.ax[1].bar(FFT.f,FFT.amps); FFT.ax[1].set_ylabel("Amplitude")
            FFT.ax[2].bar(FFT.f,FFT.phases); FFT.ax[2].set_ylabel("Phase Shift")
            
            canvas = FigureCanvasTkAgg(FFT.fig, master = Draw.signalsPlots)   
            canvas.draw() 
            canvas.get_tk_widget().pack()
            
            for i in range(0,len(FFT.idftResult)):
                FFT.idftResult[i] = round(FFT.idftResult[i].real)
            path = 'signals/IDFT_OUTPUT' + str(len(os.listdir(os.getcwd() + '/signals')) + 1) + '.txt'
            WriteFile(list(range(0,len(FFT.idftResult))),FFT.idftResult,path,0,0,len(FFT.idftResult))
        
   
    def modify():
        try:
            index = int(Draw.modifySignals.winfo_children()[1].get())
            newAmp = float(Draw.modifySignals.winfo_children()[3].get())
            newPhase = float(Draw.modifySignals.winfo_children()[5].get())
        except:
            messagebox.showerror(title = "Error",message = "Invalid Data")
            return
            
        if not(0 <= index < len(FFT.amps)):
            messagebox.showerror(title = "Error",message = "Invalid Index")
            return
        
        signal_is_drawn = (len(FFT.ax[0].lines) > 0)
            
        FFT.fig,FFT.ax = plt.subplots(nrows = 3)
        for widget in Draw.signalsPlots.winfo_children():
            widget.destroy()

        FFT.amps[index] = newAmp; FFT.phases[index] = newPhase
        
        if FFT.signal_is_present:
            if FFT.mode == "DFT":
                FFT.ax[0].stem(FFT.Y); FFT.ax[0].set_ylabel("x[n]")
            else:
                FFT.ax[0].stem(FFT.idftResult); FFT.ax[0].set_ylabel("x[n]")
                
        FFT.ax[1].bar(FFT.f,FFT.amps); FFT.ax[1].set_ylabel("Amplitude")
        FFT.ax[2].bar(FFT.f,FFT.phases); FFT.ax[2].set_ylabel("Phase Shift")
        
        canvas = FigureCanvasTkAgg(FFT.fig, master = Draw.signalsPlots)   
        canvas.draw() 
        canvas.get_tk_widget().pack()
        
    def RemoveDC():
        if(FFT.is_harmonic == 0):
            return
        
        # print(FFT.Y)

        
        FFT.amps[0] = 0
        FFT.phases[0] = 0
        
        FFT.fig,FFT.ax = plt.subplots(nrows = 3)
        for widget in Draw.signalsPlots.winfo_children():
            widget.destroy()
        
        if FFT.signal_is_present:
            
            avg = 0
            for x in FFT.Y:
                avg += x
            avg /= len(FFT.Y)
            
            for i in range(0,len(FFT.Y)):
                FFT.Y[i] -= avg
                
            
            if FFT.mode == "DFT":
                FFT.ax[0].stem(FFT.Y); FFT.ax[0].set_ylabel("x[n]")
            else:
                FFT.ax[0].stem(FFT.idftResult); FFT.ax[0].set_ylabel("x[n]")
            
        FFT.ax[1].bar(FFT.f,FFT.amps); FFT.ax[1].set_ylabel("Amplitude")
        FFT.ax[2].bar(FFT.f,FFT.phases); FFT.ax[2].set_ylabel("Phase Shift")
        
        canvas = FigureCanvasTkAgg(FFT.fig, master = Draw.signalsPlots)   
        canvas.draw() 
        canvas.get_tk_widget().pack()
        
        # print(FFT.Y)
        
            
    def Draw(x):
        
        x.deiconify()
        
        Draw.mainFrame = tk.Frame(x); Draw.mainFrame.pack()
        
        Draw.options = tk.Frame(Draw.mainFrame); Draw.options.grid(sticky = "news",row = 0)
        Draw.secondaryFrame = tk.Frame(Draw.mainFrame); Draw.secondaryFrame.grid(sticky = "news",row = 1)

        Draw.modifySignals = tk.Frame(Draw.secondaryFrame); Draw.modifySignals.grid(row = 0,sticky = "news",column = 0)
        Draw.signalsPlots = tk.Frame(Draw.secondaryFrame); Draw.signalsPlots.grid(row = 0,sticky = "news",column = 1)
        
        samlingFrequencyLabel = tk.Label(Draw.options,text = 'Samling Frequency',width = 50); samlingFrequencyLabel.grid(row = 0,column = 0)
        samlingFrequencyEntry = tk.Entry(Draw.options,width = 50); samlingFrequencyEntry.grid(row = 1,column = 0)
        modeLabel = tk.Label(Draw.options,text = 'Mode',width = 50); modeLabel.grid(row = 0,column = 1)
        modeCombox = ttk.Combobox(Draw.options,values = ['DFT','IDFT'],width = 50); modeCombox.grid(row = 1,column = 1)

        importButton =  tk.Button(Draw.options,width = 30,height = 4,text = "Import",command = ReadData); importButton.grid(row = 3,column = 0)
        modeButton =  tk.Button(Draw.options,width = 30,height = 4,text = "",command = ApplyTransformation); modeButton.grid(row = 3,column = 1)
        
        modifyLabel = tk.Label(Draw.modifySignals,font = ("Arial", 10, "bold"),text = 'Modify Component',width = 50); modifyLabel.grid(row = 0,column = 0)
        
        componentCombox = ttk.Combobox(Draw.modifySignals,values = [],width = 50); componentCombox.grid(row = 1,column = 0)
        
        newAmplitudeLabel = tk.Label(Draw.modifySignals,text = 'New Amplitude',width = 50); newAmplitudeLabel.grid(row = 2,column = 0)
        newAmplitudeEntry = tk.Entry(Draw.modifySignals,width = 50); newAmplitudeEntry.grid(row = 3,column = 0)
        
        newphaseLabel = tk.Label(Draw.modifySignals,text = 'New Phase Shift',width = 50); newphaseLabel.grid(row = 4,column = 0)
        newphaseEntry = tk.Entry(Draw.modifySignals,width = 50); newphaseEntry.grid(row = 5,column = 0)
        componentButton =  tk.Button(Draw.modifySignals,width = 30,height = 4,text = "Apply Changes",command = modify); componentButton.grid(row = 6,column = 0)
        removeDCButton =  tk.Button(Draw.modifySignals,width = 30,height = 4,text = "Remove DC Component",command = RemoveDC); removeDCButton.grid(row = 7,column = 0)
        exportButton =  tk.Button(Draw.modifySignals,width = 30,height = 4,text = "Export Signal",command = ExportSignal); exportButton.grid(row = 8,column = 0)
        

        for widget in Draw.options.winfo_children():
            widget.grid_configure(padx = 10,pady = 10)
            
        for widget in Draw.modifySignals.winfo_children():
            widget.grid_configure(padx = 5,pady = 5)
            
        for widget in Draw.signalsPlots.winfo_children():
            widget.grid_configure(padx = 5,pady = 5)
            
        x.mainloop()
    
        
    OpenOnce('FFTWindow',Draw)

def RandomSignal():
    plt.close()
    number_of_components = random.randint(1,25)
    amps = []; phases = []; freqs = []; type = [] # 0 for cos 1 for sin
    
    for i in range(0,number_of_components):
        amps.append(random.uniform(-500.0,500.0))
        phases.append(random.uniform(-2.0 * math.pi , 2.0 * math.pi))
        freqs.append(random.uniform(0.001,30.001))
        type.append(random.randint(0,1))
    
    formula = "Xa(t) = "
    for i in range(0,number_of_components):
        wave = 'cos'
        if type[i]:
            wave = 'sin'
            
        if(i and amps[i] >= 0):
            formula = formula + ' + '
        elif(amps[i] < 0):
            formula = formula + ' - '
        
        formula = formula + f"{abs(round(amps[i],2))} {wave} (2 pi {round(freqs[i], 2)} t"
        
        if(phases[i] >= 0):
            formula = formula + f" + {abs(round(phases[i],2))})"
        else:
            formula = formula + f" - {abs(round(phases[i],2))})"
            
    # print(formula)
    Y = []
    X = np.arange(0.0,10,0.001)
    
    for t in X:
        Xa_of_t = 0
        for i in range(0,number_of_components):
            if(type[i] == 0):
                Xa_of_t += amps[i] * np.cos(2 * np.pi * freqs[i] * t + phases[i])
            else:
                Xa_of_t += amps[i] * np.sin(2 * np.pi * freqs[i] * t + phases[i])
        Y.append(Xa_of_t)
    path = 'signals/signal' + str(len(os.listdir(os.getcwd() + '/signals')) + 1) + '.txt'
    WriteFile(X,Y,path,0,0,1000)
    
    with open(path,'a') as f: f.write(formula)
    fig,ax = plt.subplots()
    ax.plot(X,Y)
    fig.show()
    
def ReomveDC():
    X,Y = ReadFile()[0]
    new_Y = []
    
    def calc():
        avg_value = 0
        for val in Y:
            avg_value += val
        avg_value /= len(Y)
        ReomveDC.new_Y = []
        for val in Y:
            new_Y.append(val - avg_value)
    calc()
    
            
    def Draw(x):
        x.deiconify()
        
        Draw.mainFrame = tk.Frame(x); Draw.mainFrame.pack()
        
        Draw.Frame = tk.Frame(Draw.mainFrame); Draw.Frame.grid(sticky = "news",row = 0)

        beforeLabel = tk.Label(Draw.Frame,font = ('Trajan Pro',20,'bold'),text = 'Before', width = 50); beforeLabel.grid(row = 0)
        afterLabel = tk.Label(Draw.Frame,font = ('Trajan Pro',20,'bold'),text = 'After', width = 50); afterLabel.grid(row = 2)
                
        fig1,ax1 = plt.subplots(figsize=(5, 3)); ax1.plot(X,Y)
        fig2,ax2 = plt.subplots(figsize=(5, 3)); ax2.plot(X,new_Y)
        
        canvas = FigureCanvasTkAgg(fig1, master = Draw.Frame)   
        canvas.draw(); canvas.get_tk_widget().grid(row = 1)
        
        canvas = FigureCanvasTkAgg(fig2, master = Draw.Frame)   
        canvas.draw(); canvas.get_tk_widget().grid(row = 3)
        
        for widget in Draw.Frame.winfo_children():
            widget.grid_configure(padx = 5,pady = 5)
            
    path = 'signals/removeDC_OUTPUT' + str(len(os.listdir(os.getcwd() + '/signals')) + 1) + '.txt'
    
    WriteFile(list(range(0,len(new_Y))),new_Y,path,0,0,len(new_Y))
    
    OpenOnce('removeDCWindow',Draw)
    
def DCT():
    
    DCT.X = []
    DCT.Y = []
    DCT.new_Y = []
    DCT.N = 0
    path = 'signals/DCT_OUTPUT' + str(len(os.listdir(os.getcwd() + '/signals')) + 1) + '.txt'

    def calc():
       for k in range(0,DCT.N):
            x_of_k = 0.0
            for n in range(0,DCT.N):
                x_of_k += DCT.Y[n] * math.cos(np.pi / (4 * DCT.N) * (2 * n - 1) * (2 * k - 1))
            x_of_k *= math.sqrt(2.0 / DCT.N)
            DCT.new_Y.append(x_of_k)
            # print(DCT.new_Y)
            
    def ReadSignal():
        DCT.X,DCT.Y = ReadFile()[0]
        DCT.N = len(DCT.Y)
        calc()
        
        Draw.Frame.winfo_children()[3]['values'] = list(range(1, DCT.N + 1))
        
        beforeLabel = tk.Label(Draw.Frame2,font = ('Trajan Pro',20,'bold'), text = 'Before', width = 50); beforeLabel.grid(row = 0)
        afterLabel = tk.Label(Draw.Frame2,font = ('Trajan Pro',20,'bold'), text = 'After', width = 50); afterLabel.grid(row = 2)
                
        fig1,ax1 = plt.subplots(figsize = (5, 3)); ax1.plot(DCT.X,DCT.Y)
        fig2,ax2 = plt.subplots(figsize = (5, 3)); ax2.plot([0] * DCT.N,DCT.new_Y)
        
        canvas = FigureCanvasTkAgg(fig1, master = Draw.Frame2)   
        canvas.draw(); canvas.get_tk_widget().grid(row = 1)
        
        canvas = FigureCanvasTkAgg(fig2, master = Draw.Frame2)   
        canvas.draw(); canvas.get_tk_widget().grid(row = 3)
        
    def Export():
        if (len(DCT.new_Y)) == 0:
            messagebox.showerror(message = "There is no signal to export\n")
            return
        try:
            limit = int(Draw.Frame.winfo_children()[3].get())
        except:
            messagebox.showerror(message = "Please specify the number of elements to display\n")
            return
        DCT.new_Y = DCT.new_Y[0 : limit + 1] 
        WriteFile([0] * limit, DCT.new_Y, path, 0, 0, limit)
        
       
    
            
    def Draw(x):
        x.deiconify()
        
        Draw.mainFrame = tk.Frame(x); Draw.mainFrame.pack()
        
        Draw.Frame = tk.Frame(Draw.mainFrame); Draw.Frame.grid(sticky = "news", row = 0)
        Draw.Frame2 = tk.Frame(Draw.mainFrame); Draw.Frame2.grid(sticky = "news", row = 1)

        readSignalButton = tk.Button(Draw.Frame,font = ('Trajan Pro',20,'bold'), text = 'Read Signal', width = 30,command = ReadSignal); readSignalButton.grid(row = 0, column = 0)
        ExportButton = tk.Button(Draw.Frame,font = ('Trajan Pro',20,'bold'), text = 'Export', width = 30,command = Export); ExportButton.grid(row = 0, column = 1)
        numOfComponentsLabel = tk.Label(Draw.Frame,font = ('Trajan Pro',20,'bold'), text = 'Number of Components', width = 30); numOfComponentsLabel.grid(row = 1, column = 0)

        componentCombox = ttk.Combobox(Draw.Frame,values = [], width = 30); componentCombox.grid(row = 1,column = 1)

        
        
        for widget in Draw.Frame.winfo_children():
            widget.grid_configure(padx = 5,pady = 5)
            
        for widget in Draw.Frame2.winfo_children():
            widget.grid_configure(padx = 5,pady = 5)
            
    OpenOnce('DCTWindow',Draw)
    
def Smoothing():
    
    Smoothing.X = []
    Smoothing.Y = []
    Smoothing.newY = []
    Smoothing.newX = []
    
    def ImportSignal():

        try:
            Smoothing.X, Smoothing.Y = ReadFile()[0]
        except:
            messagebox.showerror(message = "Invalid Signal")
            
        calc()
        
        fig,ax = plt.subplots(nrows = 2,figsize = (8,5))
        
        ax[0].plot(Smoothing.X,Smoothing.Y)
        ax[1].plot(Smoothing.newX,Smoothing.newY)
        
        
        canvas = FigureCanvasTkAgg(fig, master = Draw.signal)   
        canvas.draw() 
        canvas.get_tk_widget().grid(row = 0)
        
        exportButton =  tk.Button(Draw.frame, width = 35,height = 5,text = "Export Signal",command = ExportSignal); exportButton.grid(row = 2, column = 1)
        
    def ExportSignal():
        path = ""
        try:
            path = 'signals/SMOOTHED_SIGNAL' + str(len(os.listdir(os.getcwd() + '/signals')) + 1) + '.txt'
        except:
            messagebox.showerror(message = f"Can't export the signal at this given path: \"{path}\" \n Check if there's missing files or folders in your directory")
        WriteFile(Smoothing.newX,Smoothing.newY,path,0,0,len(Smoothing.newX))
        messagebox.showinfo(message = f"Signal has been exported successfully\n with file path : \"{path}\" ")

    def calc():
        try:
            numOfPoints = int(Draw.frame.winfo_children()[1].get())
        except:
            messagebox.showerror(message = "Invalid number of points")
        if(numOfPoints < 0 or len(Smoothing.Y) - numOfPoints + 1 <= 0):
            messagebox.showerror(message = "Invalid number of points")
            return
        
        Smoothing.newY.clear()
        Smoothing.newX.clear()
        
        for i in range(0,len(Smoothing.Y) - numOfPoints + 1):
            sum = 0
            for j in range(i,i + numOfPoints):
                sum += Smoothing.Y[j]
            Smoothing.newY.append(sum / numOfPoints)
            Smoothing.newX.append(Smoothing.X[i])
       
        
        
    
    def Draw(x):
        x.deiconify()
        
        Draw.mainFrame = tk.Frame(x); Draw.mainFrame.pack()
        
        Draw.frame = tk.Frame(Draw.mainFrame); Draw.frame.grid(sticky = "news", row = 0,pady = 5)
        Draw.signal = tk.Frame(Draw.mainFrame); Draw.signal.grid(sticky = "news",row = 1, pady = 5)
        
        numOfPointsLabel = tk.Label(Draw.frame, font = ("Arial", 15),text = "Number of Points"); numOfPointsLabel.grid(row = 0)
        numOfPointsEntry = tk.Entry(Draw.frame); numOfPointsEntry.grid(row = 1)
        importButton =  tk.Button(Draw.frame, width = 35,height = 5,text = "Import Signal",command = ImportSignal); importButton.grid(row = 2)
        

        for widget in Draw.frame.winfo_children():
            widget.grid_configure(padx = 10,pady = 10)
        
        x.mainloop()

    OpenOnce("smoothingWindow",Draw)
    
def Sharpening():
        
    Sharpening.X = []
    Sharpening.Y = []
    Sharpening.newY1 = []
    Sharpening.newY2 = []
    
    def ImportSignal():

        try:
            Sharpening.X, Sharpening.Y = ReadFile()[0]
        except:
            messagebox.showerror(message = "Invalid Signal")
            
        calc()
        
        fig,ax = plt.subplots(nrows = 3,figsize = (8,5))
        
        ax[0].plot(Sharpening.X,Sharpening.Y)
        ax[1].plot(Sharpening.X,Sharpening.newY1)
        ax[2].plot(Sharpening.X,Sharpening.newY2)
        
        canvas = FigureCanvasTkAgg(fig, master = Draw.signal)   
        canvas.draw() 
        canvas.get_tk_widget().grid(row = 0)
        
        exportButton =  tk.Button(Draw.frame, width = 35,height = 5,text = "Export Signals",command = ExportSignal); exportButton.grid(row = 0, column = 1)
        
    def ExportSignal():
        path1 = ""
        path2 = ""
        
        try:
            path1 = 'signals/First Derivative' + str(len(os.listdir(os.getcwd() + '/signals')) + 1) + '.txt'
        except:
            messagebox.showerror(message = f"Can't export the signal at this given path: \"{path1}\" \n Check if there's missing files or folders in your directory")
        try:
            path2 = 'signals/Second Derivative' + str(len(os.listdir(os.getcwd() + '/signals')) + 1) + '.txt'
        except:
            messagebox.showerror(message = f"Can't export the signal at this given path: \"{path2}\" \n Check if there's missing files or folders in your directory")
        WriteFile(Sharpening.X,Sharpening.newY1,path1,0,0,len(Sharpening.X)); WriteFile(Sharpening.X,Sharpening.newY2,path2,0,0,len(Sharpening.X))
        messagebox.showinfo(message = f"Signals has been exported successfully\n with file path : \n \"{path1}\" \n \"{path2}\" ")

    def calc():
       
        Sharpening.newY1 = [Sharpening.Y[0],]
        Sharpening.newY2 = [Sharpening.Y[0],]
        
        for i in range(1,len(Sharpening.Y)):
            Sharpening.newY1.append(Sharpening.Y[i] - Sharpening.Y[i - 1])
        
        for i in range(1,len(Sharpening.Y) - 1):
            Sharpening.newY2.append(Sharpening.Y[i + 1] - 2 * Sharpening.Y[i] + Sharpening.Y[i - 1])

        Sharpening.newY2.append(Sharpening.Y[len(Sharpening.Y) - 1])
        
    
    def Draw(x):
        x.deiconify()
        
        Draw.mainFrame = tk.Frame(x); Draw.mainFrame.pack()
        
        Draw.frame = tk.Frame(Draw.mainFrame); Draw.frame.grid(sticky = "news", row = 0,pady = 5)
        Draw.signal = tk.Frame(Draw.mainFrame); Draw.signal.grid(sticky = "news",row = 1, pady = 5)
        
        importButton =  tk.Button(Draw.frame, width = 35,height = 5,text = "Import Signal",command = ImportSignal); importButton.grid(column = 0)
        
        for widget in Draw.frame.winfo_children():
            widget.grid_configure(padx = 10,pady = 10)
        
        x.mainloop()

    OpenOnce("sharpeningWindow",Draw)

def Convolution():
    
    Convolution.flag1 = 0
    Convolution.flag2 = 0
    
    Convolution.X1 = []
    Convolution.Y1 = []
    
    Convolution.X2 = []
    Convolution.Y2 = []
    
    Convolution.X3 = []
    Convolution.Y3 = []
    
    Convolution.fig,Convolution.ax = plt.subplots(nrows = 3)
    
    def ImportSignal():
        if(Convolution.flag1 and Convolution.flag2):
            return
        
        if(Convolution.flag1):
            try:
                Convolution.X2, Convolution.Y2 = ReadFile()[0]
                Convolution.flag2 = 1
            except:
                messagebox.showerror(message = "Invalid Signal")
        else:
            try:
                Convolution.X1, Convolution.Y1 = ReadFile()[0]
                Convolution.flag1 = 1
            except:
                messagebox.showerror(message = "Invalid Signal")
             
        if(Convolution.flag2):
            Convolution.ax[1].plot(Convolution.X2,Convolution.Y2)
            calc()
            exportButton =  tk.Button(Draw.frame, width = 35,height = 5,text = "Export Signals",command = ExportSignal); exportButton.grid(row = 0, column = 1)
            Convolution.ax[2].plot(Convolution.X3,Convolution.Y3)

        else:
            Convolution.ax[0].plot(Convolution.X1,Convolution.Y1)

        canvas = FigureCanvasTkAgg(Convolution.fig, master = Draw.signal)   
        canvas.draw() 
        canvas.get_tk_widget().grid(row = 0)
        
    
    def ExportSignal():
        path = ""
        try:
            path = 'signals/CONVOLUTION' + str(len(os.listdir(os.getcwd() + '/signals')) + 1) + '.txt'
        except:
            messagebox.showerror(message = f"Can't export the signal at this given path: \"{path}\" \n Check if there's missing files or folders in your directory")
        WriteFile(Convolution.X3,Convolution.Y3,path,0,0,len(Convolution.X3))
        messagebox.showinfo(message = f"Signal has been exported successfully\n with file path : \"{path}\" ")

    def calc():
        
        Convolution.X3.clear()
        Convolution.Y3.clear()
        
        Convolution.X3,Convolution.Y3 = Conv(Convolution.X1,Convolution.Y1,Convolution.X2,Convolution.Y2)
            
    
    def Draw(x):
        x.deiconify()
        
        Draw.mainFrame = tk.Frame(x); Draw.mainFrame.pack()
        
        Draw.frame = tk.Frame(Draw.mainFrame); Draw.frame.grid(sticky = "news", row = 0,pady = 5)
        Draw.signal = tk.Frame(Draw.mainFrame); Draw.signal.grid(sticky = "news",row = 1, pady = 5)
        
        importButton =  tk.Button(Draw.frame, width = 35,height = 5,text = "Import Signal",command = ImportSignal); importButton.grid(column = 0)
        
        for widget in Draw.frame.winfo_children():
            widget.grid_configure(padx = 10,pady = 10)
        
        x.mainloop()
    
    
    
    OpenOnce("convolutionWindow",Draw)
    


def Correlation():
    
    Correlation.flag1 = 0
    Correlation.flag2 = 0
    
    Correlation.X1 = []
    Correlation.Y1 = []
    
    Correlation.X2 = []
    Correlation.Y2 = []
    
    Correlation.r = []
    Correlation.normalized_r = []
    
    Correlation.fig,Correlation.ax = plt.subplots(nrows = 2,ncols = 2,figsize = (10,7))
    def DelayTime():
        try:
            fs = float(Draw.frame.winfo_children()[3].get())
        except:
            messagebox.showerror(message = "Invalid Sampling frequency !")
            
        if(len(Correlation.r) == 0):
            messagebox.showerror(message = "Calculate the correlation first")
            return
        # print(Correlation.normalized_r)
        index = 0; maximum_value = Correlation.normalized_r[0]
        for i, r in enumerate(Correlation.normalized_r):
            if(r > maximum_value):
                index = i; maximum_value = r
        # print(index)
        messagebox.showinfo(title='Delay Time',message = f'Delay time = {index * 1 / fs}')
        
    def ImportSignal():
        if(Correlation.flag1 and Correlation.flag2):
            return
        
        if(Correlation.flag1):
            try:
                Correlation.X2, Correlation.Y2 = ReadFile()[0]
                Correlation.flag2 = 1
            except:
                messagebox.showerror(message = "Invalid Signal")
        else:
            try:
                Correlation.X1, Correlation.Y1 = ReadFile()[0]
                Correlation.flag1 = 1
            except:
                messagebox.showerror(message = "Invalid Signal")
             
        if(Correlation.flag2):
            Correlation.ax[0][1].stem(Correlation.X2,Correlation.Y2); Correlation.ax[0][1].set_title('Signal 2')
            calc()
            Correlation.ax[1][0].stem(Correlation.X1,Correlation.r); Correlation.ax[1][0].set_title('Correlation')
            Correlation.ax[1][1].stem(Correlation.X2,Correlation.normalized_r); Correlation.ax[1][1].set_title('Normalized Correlation')
            exportButton =  tk.Button(Draw.frame, width = 35,height = 5,text = "Export Signals",command = ExportSignal); exportButton.grid(row = 0, column = 1)
            delayTimeButton =  tk.Button(Draw.frame, width = 35,height = 5,text = "Calculate delay time",command = DelayTime); delayTimeButton.grid(row = 0, column = 2)
            samplingFrequencyEntry = tk.Entry(Draw.frame); samplingFrequencyEntry.grid(row = 0, column = 3)
            
            for widget in Draw.frame.winfo_children(): widget.grid_configure(padx = 10,pady = 10)
        else:
            Correlation.ax[0][0].stem(Correlation.X1,Correlation.Y1); Correlation.ax[0][0].set_title('Signal 1')

        canvas = FigureCanvasTkAgg(Correlation.fig, master = Draw.signal)   
        canvas.draw() 
        canvas.get_tk_widget().grid(row = 0)
        
    
    def ExportSignal():
        path1 = ""
        path2 = ""
        
        try:
            path1 = 'signals/CORRELATION' + str(len(os.listdir(os.getcwd() + '/signals')) + 1) + '.txt'
        except:
            messagebox.showerror(message = f"Can't export the signal at this given path: \"{path1}\" \n Check if there's missing files or folders in your directory")
        try:
            path2 = 'signals/NORMALIZED_CORRELATION' + str(len(os.listdir(os.getcwd() + '/signals')) + 1) + '.txt'
        except:
            messagebox.showerror(message = f"Can't export the signal at this given path: \"{path2}\" \n Check if there's missing files or folders in your directory")
            
        WriteFile(Correlation.X1,Correlation.r,path1,0,1,len(Correlation.X1));
        WriteFile(Correlation.X1,Correlation.normalized_r,path2,0,1,len(Correlation.X1))
        messagebox.showinfo(message = f"Signals has been exported successfully\n with file path : \n \"{path1}\" \n \"{path2}\" ")

    def calc():
        
        Correlation.r.clear()
        Correlation.normalized_r.clear()
        Correlation.r,Correlation.normalized_r = Corr(Correlation.Y1,Correlation.Y2)
        
    
    def Draw(x):
        x.deiconify()
        
        Draw.mainFrame = tk.Frame(x); Draw.mainFrame.pack()
        
        Draw.frame = tk.Frame(Draw.mainFrame); Draw.frame.grid(sticky = "news", row = 0,pady = 5)
        Draw.signal = tk.Frame(Draw.mainFrame); Draw.signal.grid(sticky = "news",row = 1, pady = 5)
        
        importButton =  tk.Button(Draw.frame, width = 35,height = 5,text = "Import Signal",command = ImportSignal); importButton.grid(column = 0)
        
        for widget in Draw.frame.winfo_children():
            widget.grid_configure(padx = 10,pady = 10)
        
        x.mainloop()
    
    
    
    OpenOnce("correlationbWindow",Draw)
    

def Classification():
    Classification.number_of_classes = -1
    Classification.number_of_present_classes = 0
    Classification.dic = {}
    
    # dictionary it's key is the index of the given class and the value represent a list of tuples each tuple represent a file the tuple is always a pair the first element is the grid of columns or the signals that we deal with and the second element is the 3 bits meta data
    Classification.folders = {}
    
    # dictionary as for the key it represent the index of the signal as for the value it represent a signal. the one element represent an averged signal for the signals within it's class that correspond to the index (key)
    Classification.averaged_signals = {}
    
    Classification.flag1 = 0
    Classification.flag2 = 0
    
    Classification.Y = []
    
    Classification.signal_fig, Classification.signal_ax = plt.subplots(nrows = 1, ncols = 1,figsize = (15,3))
    Classification.fig,Classification.ax = plt.subplots(nrows = 1,ncols = 2,figsize = (15,3))
    
    
    
    
    def ImportClass(index):
        Classification.folders.update({index:ReadDirectory(1,0)})
        
        if Classification.averaged_signals.get(index) == None:
            Classification.number_of_present_classes +=1
            
        Classification.averaged_signals.update({index: []})
        
            
        
        for file in Classification.folders[index]:
            Classification.averaged_signals[index].append(file[0][0])
            
        Classification.averaged_signals[index] = average_signals(Classification.averaged_signals[index])
        
        Classification.ax[index].plot(Classification.averaged_signals[index])
        
        canvas = FigureCanvasTkAgg(Classification.fig, master = Draw.signal2); canvas.draw(); canvas.get_tk_widget().grid(row = 0)
        
        if Classification.number_of_present_classes == Classification.number_of_classes:
            classificationButton =  tk.Button(Draw.frame, width = 35,height = 5,text = "Start Classification", command = calc); classificationButton.grid(row = 0, column = 3)

        
    def ImportClasses():   
        try:
            Classification.number_of_classes = int(Draw.frame.winfo_children()[2].get())
        except:
            messagebox.showerror(message = "Invalid number of classes \n it should be positive integer")
            return
        if Classification.number_of_classes < 0:
            messagebox.showerror(message = "Only Positive integers ..!\n")
            return
        
        for widget in Draw.subFrame.winfo_children():
            widget.destroy()
            
        Draw.Classes_buttons.clear()
        Classification.averaged_signals.clear()
        Classification.folders.clear()
        Classification.number_of_present_classes = 0
        
        Classification.fig,Classification.ax = plt.subplots(nrows = 1,ncols = Classification.number_of_classes, figsize = (15,3))
        for i in range(0, Classification.number_of_classes):
            Classification.ax[i].plot([]); Classification.ax[i].set_title(f"Class {i + 1}")
            Draw.Classes_buttons.append(tk.Button(Draw.subFrame, width = 15,height = 5, text = f"Import Class {i + 1}",command = lambda idx = i: ImportClass(idx)))
            Draw.Classes_buttons[i].grid(row = 0, column = i)
        
        canvas = FigureCanvasTkAgg(Classification.fig, master = Draw.signal2)  
        canvas.draw() 
        canvas.get_tk_widget().grid(row = 0)
        
        
    def ImportSignal():
        
        if(Classification.flag1 and Classification.flag2 == Classification.number_of_classes):
            return
        
        if(Classification.flag1 == 0):
            try:
                Classification.Y = ReadFile(num_of_columns = 1,meta_data_bits = 0)[0][0]
                # print(Classification.Y)
                Classification.flag1 = 1
            except:
                messagebox.showerror(message = "Invalid Signal \n the format of this file must just be one column of samples without meta data bits")
            try:
                Classification.signal_ax.plot(Classification.Y); Classification.signal_ax.set_title("Signal to be Classified")
                canvas = FigureCanvasTkAgg(Classification.signal_fig, master = Draw.signal1)   
                canvas.draw() 
                canvas.get_tk_widget().grid(row = 0)
            except:
                messagebox.showerror(message = "Can't Draw this signal.!")
            numOfClassesButton = tk.Button(Draw.frame, width = 35,height = 5,text = "Import Classes",command = ImportClasses); numOfClassesButton.grid(row = 0,column = 1)
            numOfClassesEntry = tk.Entry(Draw.frame); numOfClassesEntry.grid(row = 0,column = 2)
            
            for widget in Draw.frame.winfo_children():
                widget.grid_configure(padx = 10,pady = 10)
                
        elif Classification.number_of_classes == -1:
            messagebox.showerror(message = "please specify the number of classes !")
            return
             
       
    
    def calc():
        higest = None
        index = 0
        for i in range(0, Classification.number_of_classes):
            score = Corr(Classification.Y,Classification.averaged_signals[i])[1][0]
            
            if(higest == None or score > higest):
                higest = score; index = i
        messagebox.showinfo(message = f"the signal belongs to class {index + 1}")
        
    def Draw(x):
        
        # layer 0
        x.deiconify()
        
        # layer 1
        Draw.mainFrame = tk.Frame(x); Draw.mainFrame.pack() 
        
        # layer 2
        Draw.frame = tk.Frame(Draw.mainFrame); Draw.frame.grid(sticky = "news", row = 0,pady = 5)
        Draw.signal1 = tk.Frame(Draw.mainFrame); Draw.signal1.grid(sticky = "news",row = 1, pady = 5)
        Draw.signal2 = tk.Frame(Draw.mainFrame); Draw.signal2.grid(sticky = "news",row = 2, pady = 5)
        
        # layer 3
        Draw.subFrame = tk.Frame(Draw.signal2); Draw.subFrame.grid(sticky = "news",row = 1, pady = 5)
        
        importButton =  tk.Button(Draw.frame, width = 35,height = 5,text = "Import Signal",command = ImportSignal); importButton.grid(row = 0, column = 0)
        Draw.Classes_buttons = []

        for widget in Draw.frame.winfo_children():
            widget.grid_configure(padx = 10,pady = 10)
        
        x.mainloop()
    
    
    
    OpenOnce("classificationWindow",Draw)


def FIR_GUI():
    
    FIR_GUI.is_lpf = 0
    
    FIR_GUI.filterType = None
    FIR_GUI.fs = None
    FIR_GUI.stopBandAttenuation = None
    FIR_GUI.f1 = None
    FIR_GUI.f2 = None
    FIR_GUI.transitionWidth = None
    FIR_GUI.fir_object = None
    FIR_GUI.signal_fig, FIR_GUI.signal_ax = plt.subplots(nrows = 1, ncols = 1,figsize = (15,4.5))
    

        
    def Clear_info():
        FIR_GUI.filterType = FIR_GUI.fs = FIR_GUI.stopBandAttenuation = FIR_GUI.f1 = FIR_GUI.f2 = FIR_GUI.transitionWidth = FIR_GUI.fir_object = None
        FIR_GUI.signal_fig.clf(); FIR_GUI.signal_ax.clear()
        FIR_GUI.signal_fig, FIR_GUI.signal_ax = plt.subplots(nrows = 1, ncols = 1,figsize = (15,4.5))
        
        if len(Draw.signal.winfo_children()):
            Draw.signal.winfo_children()[0].destroy()
        for i,widget in enumerate(Draw.frame.winfo_children()):
            if i != 0:
                widget.destroy()
        
    
    def ImportFilterSpecifications():
        
        Clear_info()
        
        format_message = '''
        the format of the file must be on following format :
        
        FilterType = {Band pass}{Band stop}{Low pass}{High pass}
        FS = x
        StopBandAttenuation = x
        F1 = x
        F2 = x # <- only if band pass or band stop
        TransitionBand = x
        '''
        messagebox.showwarning(message = format_message)
        
        
        try:
            filename = filedialog.askopenfilename()
            with open(filename,'r') as f:
                temp = f.readline().rstrip('\n').split(' ')
                
                FIR_GUI.filterType = temp[2].lower() + temp[3].lower()#
                
                temp = f.readline().rstrip('\n').split(' ')
                
                FIR_GUI.fs = float(temp[2])#
                
                temp = f.readline().rstrip('\n').split(' ')
                
                FIR_GUI.stopBandAttenuation = float(temp[2])#
                
                temp = f.readline().rstrip('\n').split(' ')
                
                FIR_GUI.f1 = float(temp[2])#
                
                if(FIR_GUI.filterType == 'bandpass' or FIR_GUI.filterType == 'bandstop'):
                    temp = f.readline().rstrip('\n').split(' ')
                    FIR_GUI.f2 = float(temp[2])#
                    
                temp = f.readline().rstrip('\n').split(' ')

                FIR_GUI.transitionWidth = float(temp[2])
        except:
            messagebox.showerror(message = format_message)
            
            

        FIR_GUI.fir_object = FIR(FIR_GUI.f1,FIR_GUI.fs,
                                FIR_GUI.stopBandAttenuation,FIR_GUI.transitionWidth,
                                FIR_GUI.f2,FIR_GUI.filterType)
        
        
        
        X = [0,] 
        Y = [FIR_GUI.fir_object.get_hd_of_n(0)]
        for n in range(1, int(((FIR_GUI.fir_object.N) - 1) / 2) + 1):
            X.append(n)
            X.insert(0,-n)
            temp_y = FIR_GUI.fir_object.get_h_of_n(n)
            Y.append(temp_y)
            Y.insert(0,temp_y)
        
        is_band_pass_or_stop = (FIR_GUI.filterType == 'bandpass' or FIR_GUI.filterType == 'bandstop')
        
        filterTypeLabel = tk.Label(Draw.frame,font = ('Arial',25),text = f"Filter Type = {FIR_GUI.filterType}"); filterTypeLabel.grid(row = 1,column = 0)
        stopBandAttenuationLabel = tk.Label(Draw.frame,font = ('Arial',25),text = f"Stop Band Attenuation = {FIR_GUI.stopBandAttenuation}"); stopBandAttenuationLabel.grid(row = 1,column = 1)
        fsLabel = tk.Label(Draw.frame,font = ('Arial',25),text = f"Sampling frequency = {FIR_GUI.fs}"); fsLabel.grid(row = 2,column = 0)
        f1Label = tk.Label(Draw.frame,font = ('Arial',25),text = f"cut off frequency {1 if is_band_pass_or_stop else ''} = {FIR_GUI.f1}"); f1Label.grid(row = 2,column = 1)
        if is_band_pass_or_stop:
            f2Label = tk.Label(Draw.frame,font = ('Arial',25),text = f"cut off frequency 2 = {FIR_GUI.f2}"); f2Label.grid(row = 2,column = 2)
        transitionWidthLabel = tk.Label(Draw.frame,font = ('Arial',25),text = f"Transition Width = {FIR_GUI.transitionWidth}"); transitionWidthLabel.grid(row = 3,column = 0)
        windoWLabel = tk.Label(Draw.frame,font = ('Arial',25),text = f"Window is {FIR_GUI.fir_object.window}"); windoWLabel.grid(row = 3,column = 1)

        path = 'signals/FIR_FILTER' + str(len(os.listdir(os.getcwd() + '/signals')) + 1) + '.txt'
        exportButton = H.Button(root = Draw.frame,name = "Export Filter", position = (0,1), on_click = lambda: WriteFile(X,Y,path,0,0,FIR_GUI.fir_object.N))

        FIR_GUI.signal_ax.stem(X,Y); FIR_GUI.signal_ax.set_title(f"{FIR_GUI.fir_object.filter_type}")
        canvas = FigureCanvasTkAgg(FIR_GUI.signal_fig, master = Draw.signal)   
        canvas.draw() 
        canvas.get_tk_widget().pack()
            
    
    def Draw(x):
        # layer 0
        x.deiconify()
        
        # layer 1
        
        Draw.mainFrame = tk.Frame(x); Draw.mainFrame.pack()
        
        # layer 2
        Draw.frame = tk.Frame(Draw.mainFrame); Draw.frame.grid(sticky = "news", row = 0,column = 0,pady = 5)
        Draw.signal = tk.Frame(Draw.mainFrame); Draw.signal.grid(sticky = "news",row = 1,column = 0, pady = 5)

        
        importButton = H.Button(root = Draw.frame, name = "Import Filter Specifications", position = (0,0), on_click = ImportFilterSpecifications)
        x.mainloop()

    
    OpenOnce("FIRWindow",Draw)


        
def FileManipulation():
    
    FileManipulation.isPaddingWindowOpen = 0
    
    def Padding(): # generalization possibility : YES
        
        if(FileManipulation.isPaddingWindowOpen == 1):
            messagebox.showerror(message = "Window ALready Open!\n")
            return
        FileManipulation.isPaddingWindowOpen = 1
        def FolderOption():
            signals = ReadDirectory(meta_data_bits = 1)
            signals = FILE.SignalsPadding(signals)
        
        def FileOption():
            FileOption.cnt = 0
            FileOption.signals = []
            
            def ReadSignal():
                signal = ReadFile(num_of_columns = 2,meta_data_bits = with_meta_data.get())
                
                if signal == None:
                    return
                
                FileOption.signals.append(signal) 
                FileOption.cnt += 1
                
                dialog_window.mainFrame.winfo_children()[0]['text'] = f"Number of Added Signals :\t{FileOption.cnt}"
            
            dialog_window = H.DialogButtons(window = fileManipulationWindow,height = 200)
            
            label = tk.Label(dialog_window.mainFrame,text = f"Number of Added Signals :\t{FileOption.cnt}"); label.grid(row = 0,column = 0)
            importButton = H.Button(root = dialog_window.mainFrame, position = (1, 0), width = 35,height = 5, pady = 5, name = "import signal",on_click = ReadSignal)
            
            with_meta_data = tk.BooleanVar(); checkbox = tk.Checkbutton(dialog_window.mainFrame,text = "with meta data bits ?", variable = with_meta_data); checkbox.grid(row = 1,column = 1)
            applyButton = H.Button(root = dialog_window.mainFrame, position = (2, 0), width = 35,height = 5, pady = 5, name = "Apply Padding", on_click = lambda: FILE.SignalsPadding(FileOption.signals, export_file = 1) if (FileOption.cnt >= 2) else messagebox.showerror(message = "at least two signals must be added\n"))
            cancelButton = H.Button(root = dialog_window.mainFrame, position = (2, 1), width = 35,height = 5, pady = 5, name = "Cancel", on_click = lambda: dialog_window.dialog.destroy())
            
            
        def close_dialog():
            FileManipulation.isPaddingWindowOpen = 0
        
        dialog_window = H.DialogButtons(window = fileManipulationWindow)
        dialog_window.dialog.protocol("WM_DELETE_WINDOW", func = close_dialog)
        
        btn1 = H.Button(root = dialog_window.mainFrame, position = (0, 0), width = 35,height = 5, pady = 5, name = "SELECT FOLDER", on_click = FolderOption)
        btn2 = H.Button(root = dialog_window.mainFrame, position = (0, 1), width = 35,height = 5, pady = 5, name = "SELECT FILES", on_click = FileOption)
        btn3 = H.Button(root = dialog_window.mainFrame, position = (0, 2), width = 35,height = 5, pady = 5, name = "Cancel", on_click = lambda: dialog_window.dialog.destroy())
        

    def Draw(x):
    
        # layer 0
        x.deiconify()

        
        # layer 1
        Draw.mainFrame = tk.Frame(x); Draw.mainFrame.pack() 
        
        # layer 2
        Draw.frame = tk.Frame(Draw.mainFrame); Draw.frame.grid(sticky = "news", row = 0,pady = 5)
        
        cartesianConversionButton =  tk.Button(Draw.frame, width = 35,height = 5,text = "Convert to Cartesian Form",command = FILE.FromPolarToCartesian); cartesianConversionButton.grid(row = 0, column = 0)
     
        polarConversionButton = tk.Button(Draw.frame, width = 35,height = 5,text = "Convert to Polar Form",command = FILE.FromCartesianToPolar); polarConversionButton.grid(row = 1, column = 0)
                
        signalsPadding = H.Button(root = Draw.frame, name = "Signals Padding",position = (2,0),on_click = Padding)
        
        for widget in Draw.frame.winfo_children():
            widget.grid_configure(padx = 10,pady = 10)
        x.mainloop()
    
    OpenOnce("fileManipulationWindow",Draw)

def Resampling_GUI():

    Resampling_GUI.L = None
    Resampling_GUI.M = None
    
    Resampling_GUI.file = None
    
    Resampling_GUI.signal_fig, Resampling_GUI.signal_ax = plt.subplots(nrows = 1, ncols = 2,figsize = (15,4.5))
    
    def Clear_info():
        Resampling_GUI.L = Resampling_GUI.M = Resampling_GUI.file = None
        Resampling_GUI.signal_fig.clf()
        
        Resampling_GUI.signal_fig, Resampling_GUI.signal_ax = plt.subplots(nrows = 1, ncols = 2,figsize = (15,4.5))
        

        
        if len(Draw.signals.winfo_children()):
            Draw.signals.winfo_children()[0].destroy()
            
        if len(Draw.frame.winfo_children()) > 6:
            for i in range(5,len(Draw.frame.winfo_children())):
                Draw.signals.winfo_children()[i].destroy()
          

    def filter(X,Y):
        
        X2,Y2 = ReadFile(num_of_columns = 2,meta_data_bits = 1)[0]
        Resampling_GUI.file[0][0],Resampling_GUI.file[0][1] = Conv(X,Y,X2,Y2)
        
 
       
    def RUN():
        try:
            Resampling_GUI.L = int(Draw.frame.winfo_children()[2].get())
            Resampling_GUI.M = int(Draw.frame.winfo_children()[4].get())
        except:
            messagebox.showerror(message = "Invalid Values for L or M\n")
            return
        
        Resampling_GUI.file[0][0],Resampling_GUI.file[0][1] = Resampling(Resampling_GUI.L, Resampling_GUI.M,Resampling_GUI.file[0][0], Resampling_GUI.file[0][1])
        
        
        path = 'signals/ReSampled_Signal' + str(len(os.listdir(os.getcwd() + '/signals')) + 1) + '.txt'

        H.Button(Draw.frame,"export Signal",(1,0),on_click = lambda : WriteFile(Resampling_GUI.file[0][0],Resampling_GUI.file[0][1], path, 0,0,len(Resampling_GUI.file[0][1]), meta_data_bits = 1))
        
        Draw.signals.winfo_children()[0].destroy()
        Resampling_GUI.signal_ax[1].stem(Resampling_GUI.file[0][0],Resampling_GUI.file[0][1]); Resampling_GUI.signal_ax[1].set_title("Resampled Signal")
        canvas = FigureCanvasTkAgg(Resampling_GUI.signal_fig, master = Draw.signals)   
        canvas.draw() 
        canvas.get_tk_widget().pack()
        
        
        

    def ReadSignal():

        Clear_info()
        
        Resampling_GUI.file = ReadFile(num_of_columns = 2,meta_data_bits = 1)
        
        try:
            Resampling_GUI.L = int(Draw.frame.winfo_children()[2].get())
            Resampling_GUI.M = int(Draw.frame.winfo_children()[4].get())
        except:
            messagebox.showerror(message = "Invalid Values for L or M\n")
            return
        
        H.Button(Draw.frame,"Convolution with low pass filter",(0,6), on_click = lambda: filter(Resampling_GUI.file[0][0],Resampling_GUI.file[0][1]))
        H.Button(Draw.frame,"Resample!",(0,7), on_click = RUN)
        
        Resampling_GUI.signal_ax[0].stem(Resampling_GUI.file[0][0],Resampling_GUI.file[0][1]); Resampling_GUI.signal_ax[0].set_title("Original Signal")
        
        canvas = FigureCanvasTkAgg(Resampling_GUI.signal_fig, master = Draw.signals)   
        canvas.draw() 
        canvas.get_tk_widget().pack()
        
        
        
    def Draw(x):
        # layer 0
        x.deiconify()
        
        # layer 1
        
        Draw.mainFrame = tk.Frame(x); Draw.mainFrame.pack()
        
        # layer 2
        Draw.frame = tk.Frame(Draw.mainFrame); Draw.frame.grid(sticky = "news", row = 0,column = 0,pady = 5)
        Draw.signals = tk.Frame(Draw.mainFrame); Draw.signals.grid(sticky = "news",row = 1,column = 0, pady = 5)

        importButton = H.Button(root = Draw.frame,name = "Import Signal", position = (0,0), on_click = ReadSignal)
        LCoefficientLabel = tk.Label(Draw.frame,font = ('Arial',20),text = "L Value :"); LCoefficientLabel.grid(row = 0,column = 1, padx = 10)
        LEntry = tk.Entry(Draw.frame); LEntry.grid(row = 0, column = 2)
        
        MCoefficientLabel = tk.Label(Draw.frame,font = ('Arial',20),text = "M Value :"); MCoefficientLabel.grid(row = 0,column = 3, padx = 10)
        MEntry = tk.Entry(Draw.frame); MEntry.grid(row = 0, column = 4)
        
        x.mainloop()
    
    OpenOnce("resamplingWindow",Draw)

def ECG(): # Not completed yet
    ECG.list_of_classes = []
    ECG.miniF = None
    ECG.maxiF = None
    ECG.old_fs = None
    ECG.fs = None
    
    ECG.currentSignal = []
    
    def add_class():
        ECG.list_of_classes.append(ReadDirectory(1,0))
        
        Draw.frame.winfo_children()[1]['text'] = f"Number of present Classes = {len(ECG.list_of_classes)}"
    
    def ResamplingCurrentSignal():
        dialog = H.DialogButtons(ECGClassificationWindow,height = 180)
        newFrequecyLabel = tk.Label(dialog.mainFrame,font = ('Arial', 20), text = "new Frequency :"); newFrequecyLabel.grid(row = 0,column = 0)
        newFrequecyEntry = tk.Entry(dialog.mainFrame); newFrequecyEntry.grid(row = 0,column = 1)
        
        try:
           ECG.old_fs = float(Draw.frame.winfo_children()[3].get())
        except:
            messagebox.showerror(message = "Invalid Sampling frequency of the original signal")
            dialog.dialog.destroy()
            return
        
        upSamplingLabel = tk.Label(dialog.mainFrame,font = ('Arial', 20), text = "M :"); upSamplingLabel.grid(row = 1,column = 0,pady = 5)
        newFrequecyEntry = tk.Entry(dialog.mainFrame); newFrequecyEntry.grid(row = 1,column = 1)
        
        downSamplingLabel = tk.Label(dialog.mainFrame,font = ('Arial', 20), text = "L :"); downSamplingLabel.grid(row = 1,column = 2)
        newFrequecyEntry = tk.Entry(dialog.mainFrame); newFrequecyEntry.grid(row = 1,column = 3)
        
        H.Button(dialog.mainFrame,"Apply")
        
    def add_signal():
        ECG.currentSignal = ReadFile(num_of_columns = 1,meta_data_bits = 0)[0][0]
        H.Button(Draw.frame,"Resample the signal",(4,1),on_click = ResamplingCurrentSignal,width = 30)
           
    def Draw(x):
        # layer 0
        x.deiconify()
        
        # layer 1
        
        Draw.mainFrame = tk.Frame(x); Draw.mainFrame.pack()
        
        # layer 2
        Draw.frame = tk.Frame(Draw.mainFrame); Draw.frame.grid(sticky = "news", row = 0,column = 0,pady = 5)
        Draw.signals = tk.Frame(Draw.mainFrame); Draw.signals.grid(sticky = "news",row = 1,column = 0, pady = 5)

        numberOfClassesLabel = tk.Label(Draw.frame,font = ('Arial',20), text = "Number of present Classes = 0"); numberOfClassesLabel.grid(row = 0,column = 0, padx = 10)
        H.Button(Draw.frame,"Import Class",(0,1),on_click = add_class)
        
        fsLabel = tk.Label(Draw.frame, font = ('Arial',20), text = "Sampling Frequency"); fsLabel.grid(row = 1,column = 0) 
        fsEntry = tk.Entry(Draw.frame); fsEntry.grid(row = 0, column = 2); fsEntry.grid(row = 1,column = 1)
        
        
        miniFLabel = tk.Label(Draw.frame, font = ('Arial',20), text = "minimum Frequency"); miniFLabel.grid(row = 2,column = 0) 
        miniFEntry = tk.Entry(Draw.frame); miniFEntry.grid(row = 2, column = 1)
        
        
        maxiFLabel = tk.Label(Draw.frame, font = ('Arial',20), text = "maximum Frequency"); maxiFLabel.grid(row = 3, column = 0) 
        maxiFEntry = tk.Entry(Draw.frame); maxiFEntry.grid(row = 3, column = 1)
        
        H.Button(Draw.frame,"Import signal",(4,0),on_click = add_signal)
        
        
        x.mainloop()

    OpenOnce("ECGClassificationWindow",Draw)

    


mainWindow.geometry("1300x700")
readyInputsWindow.geometry("1300x700")
SignalOperationsWindow.geometry("1300x700")
quantizationWindow.geometry("1300x700")
FFTWindow.geometry("1300x700")
randomSignalWindow.geometry("1300x700")
removeDCWindow.geometry("1300x700")
DCTWindow.geometry("1300x700")
smoothingWindow.geometry("1300x700")
sharpeningWindow.geometry("1300x700")
convolutionWindow.geometry("1300x700")
correlationbWindow.geometry("1300x700")
classificationWindow.geometry("1300x700")
fileManipulationWindow.geometry("1300x700")
FIRWindow.geometry("1300x700")
resamplingWindow.geometry("1300x700")
ECGClassificationWindow.geometry("1300x700")

mainWindow.state("zoomed")



mainWindow.attributes('-topmost', False)



mainFrame = tk.Frame(mainWindow); mainFrame.pack()

signalOptionsFrame = tk.LabelFrame(mainFrame,text = 'Signal Options'); signalOptionsFrame.grid(sticky = "news",row = 0, column = 0)

fileOptionsFrame = tk.LabelFrame(mainFrame,text = 'File Options'); fileOptionsFrame.grid(sticky = "news",row = 1, column = 0)


    
readFromFileOption = tk.Button(signalOptionsFrame,text ="Read from file",width = 30, height = 6,command = display); readFromFileOption.grid(row = 0,column = 0)
readyDataOption = tk.Button(signalOptionsFrame,text="Ready inputs",width = 30,height = 6,command = ReadyInputs); readyDataOption.grid(row = 0,column = 1)
signalOperationsOption = tk.Button(signalOptionsFrame,text="Operations",width = 30,height = 6,command = SignalOperations); signalOperationsOption.grid(row = 0,column = 2)
quantizationOption = tk.Button(signalOptionsFrame,text="Quantization",width = 30,height = 6,command = Quantization); quantizationOption.grid(row = 0,column = 3)
FFTOption = tk.Button(signalOptionsFrame,text="Fast Fourier Transformation",width = 30,height = 6,command = FFT); FFTOption.grid(row = 1,column = 0)
createSignalOption = tk.Button(signalOptionsFrame,text="Create Random Signal",width = 30,height = 6,command = RandomSignal); createSignalOption.grid(row = 1,column = 1)
removeDCOption = tk.Button(signalOptionsFrame,text="Remove DC Component",width = 30,height = 6,command = ReomveDC); removeDCOption.grid(row = 1,column = 2)
DCTOption = tk.Button(signalOptionsFrame,text="DCT",width = 30,height = 6,command = DCT); DCTOption.grid(row = 1,column = 3)
smoothingOption = tk.Button(signalOptionsFrame,text="Smoothing",width = 30,height = 6,command = Smoothing); smoothingOption.grid(row = 2,column = 0)
sharpeningOption = tk.Button(signalOptionsFrame,text = "Sharpening",width = 30,height = 6,command = Sharpening); sharpeningOption.grid(row = 2,column = 1)
convolutionOption = tk.Button(signalOptionsFrame,text = "Convolution",width = 30,height = 6,command = Convolution); convolutionOption.grid(row = 2,column = 2)
ECGClassificationOption = tk.Button(signalOptionsFrame,text = "ECG",width = 30,height = 6,command = ECG); ECGClassificationOption.grid(row = 2,column = 3)

correlationbOption = tk.Button(signalOptionsFrame,text = "Correlation",width = 30,height = 6,command = Correlation); correlationbOption.grid(row = 3,column = 0)
classificationOption = tk.Button(signalOptionsFrame,text = "Classify a Signal",width = 30, height = 6, command = Classification); classificationOption.grid(row = 3,column = 1)
FIROption = tk.Button(signalOptionsFrame,text = "FIR",width = 30, height = 6, command = FIR_GUI); FIROption.grid(row = 3,column = 2)
resamplingOption = tk.Button(signalOptionsFrame,text = "Resampling",width = 30, height = 6, command = Resampling_GUI); resamplingOption.grid(row = 3,column = 3)


fileManipulationButton = tk.Button(fileOptionsFrame,text = "File Manipulation",width = 30, height = 6,command = FileManipulation); fileManipulationButton.grid(row = 0,column = 0)

mainWindow.mainloop()