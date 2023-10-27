import numpy as np
import pickle
import pandas as pd
import streamlit as st
from PIL import Image
# import random

# import functions from the utils module
from utils import update_buffer, get_last_data, compute_band_powers
from pylsl import StreamInlet, resolve_byprop

# 0. Apply some custom styling! Change colors, images, etc.
st.markdown(
     f"""
     <style>
      
    .stApp {{
            background: url("https://www.thoughtco.com/thmb/g8h6NnWWWVkm-KXNBgMx-0Edd2U=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/GettyImages_482194715-56a1329e5f9b58b7d0bcf666.jpg");
            background-size: cover
        }}

    .css-1n76uvr {{
        background-color: rgba(255, 255, 255, 0.6);
        padding: 100px;
        border-radius: 30px
    }}
    
    *{{
        color: black;
        text-align: center
    }}
    
</style>
     """,
     unsafe_allow_html=True
 )


BUFFER_LENGTH = 5
EPOCH_LENGTH = 1
OVERLAP_LENGTH = 0
SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH
INDEX_CHANNEL = [0]

# 1. LOAD YOUR MODEL
with open("my_model.pkl", 'rb') as file:
    clf = pickle.load(file)

with st.container():
    
    # 2. WRITE ABOUT WHAT YOU DID
    st.header('My BioBinary Project')
    st.markdown('*A Synaptech Project*')
    st.markdown('🌟 **Team:** Put your team names here!🌟')
    st.markdown("I am classifying between eyes open and eyes closed using a decision tree.")
    
    st.markdown("Here I'm going to type a bit about decision trees and the hyperparameter \
                tuning I tried!")
    
    # 3. INCLUDE IMAGES OF PLOTS (EX: FOURIER TRANSFORM, CORR MATRIX, OR PICTURES
    #    OF YOU WEARING EEG, ANYTHING REALLY LOL)
    # image = Image.open('your_image.png')
    # st.image(image)
    
    if st.button('Go'):

        print('Looking for an EEG stream...')
        streams = resolve_byprop('type', 'EEG', timeout=2)
        if len(streams) == 0:
            raise RuntimeError('Can\'t find EEG stream.')
        else:
            print('Found it!')
            print(streams)
            
        # Set active EEG stream to inlet and apply time correction
        print("Start acquiring data")
        inlet = StreamInlet(streams[0], max_chunklen=12)
        eeg_time_correction = inlet.time_correction()

        # Get the stream info
        info = inlet.info()
        fs = int(info.nominal_srate())

        # Initialize raw EEG data buffer
        eeg_buffer = np.zeros((int(fs * BUFFER_LENGTH), 1))
        filter_state = None  # for use with the notch filter

        # Compute the number of epochs in "buffer_length"
        n_win_test = int(np.floor((BUFFER_LENGTH - EPOCH_LENGTH) /
                                    SHIFT_LENGTH + 1))

        # Initialize the band power buffer (for plotting)
        # bands will be ordered: [delta, theta, alpha, beta]
        band_buffer = np.zeros((n_win_test, 4))

        print('Press Ctrl-C in the console to break the while loop.')

        while True:
            # Obtain EEG data from the LSL stream
            eeg_data, timestamp = inlet.pull_chunk(
                timeout=1, max_samples=int(SHIFT_LENGTH * fs))

            # Only keep the channel we're interested in
            ch_data = np.array(eeg_data)[:, INDEX_CHANNEL]

            # Update EEG buffer with the new data
            eeg_buffer, filter_state = update_buffer(
                eeg_buffer, ch_data, notch=True,
                filter_state=filter_state)

            # Get newest samples from the buffer
            data_epoch = get_last_data(eeg_buffer,
                                                EPOCH_LENGTH * fs)

            # Compute band powers
            band_powers = compute_band_powers(data_epoch, fs)
            band_buffer, _ = update_buffer(band_buffer,
                                                    np.asarray([band_powers]))
            
            # testing
            # band_powers = [12.4, 12.3, 9.8, 11.2]
            # band_powers = [random.uniform(0, 20) for _ in band_powers]

            # Build dataframe
            data = {
                'alpha': [band_powers[2]],
                'beta': [band_powers[3]],
                'theta': [band_powers[1]],
                'delta': [band_powers[0]]
            }
            
            df = pd.DataFrame(data)
            
            # Predict
            result = clf.predict(df.values)
            
            # 4. CUSTOMIZE WHAT HAPPENS AFTER A PREDICTION IS MADE!
            if result[0] == 1:
                st.markdown('Eyes are open!')
                # image = Image.open('your_image.png')
                # st.image(image)
            
            else:
                st.markdown('Eyes are closed!')
                # image = Image.open('your_image.png')
                # st.image(image)

