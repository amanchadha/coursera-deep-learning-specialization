"""
File: midi.py
Author: Addy771
Description: 
A script which converts MIDI files to WAV and optionally to MP3 using ffmpeg. 
Works by playing each file and using the stereo mix device to record at the same time
"""


import pyaudio  # audio recording
import wave     # file saving
import pygame   # midi playback
import fnmatch  # name matching
import os       # file listing


#### CONFIGURATION ####

do_ffmpeg_convert = True    # Uses FFmpeg to convert WAV files to MP3. Requires ffmpeg.exe in the script folder or PATH
do_wav_cleanup = True       # Deletes WAV files after conversion to MP3
sample_rate = 44100         # Sample rate used for WAV/MP3
channels = 2                # Audio channels (1 = mono, 2 = stereo)
buffer = 1024               # Audio buffer size
mp3_bitrate = 128           # Bitrate to save MP3 with in kbps (CBR)
input_device = 1            # Which recording device to use. On my system Stereo Mix = 1



# Begins playback of a MIDI file
def play_music(music_file):

    try:
        pygame.mixer.music.load(music_file)
        
    except pygame.error:
        print ("Couldn't play %s! (%s)" % (music_file, pygame.get_error()))
        return
        
    pygame.mixer.music.play()



# Init pygame playback
bitsize = -16   # unsigned 16 bit
pygame.mixer.init(sample_rate, bitsize, channels, buffer)

# optional volume 0 to 1.0
pygame.mixer.music.set_volume(1.0)

# Init pyAudio
format = pyaudio.paInt16
audio = pyaudio.PyAudio()



try:

    # Make a list of .mid files in the current directory and all subdirectories
    matches = []
    for root, dirnames, filenames in os.walk("./"):
        for filename in fnmatch.filter(filenames, '*.mid'):
            matches.append(os.path.join(root, filename))
            
    # Play each song in the list
    for song in matches:

        # Create a filename with a .wav extension
        file_name = os.path.splitext(os.path.basename(song))[0]
        new_file = file_name + '.wav'

        # Open the stream and start recording
        stream = audio.open(format=format, channels=channels, rate=sample_rate, input=True, input_device_index=input_device, frames_per_buffer=buffer)
        
        # Playback the song
        print("Playing " + file_name + ".mid\n")
        play_music(song)
        
        frames = []
        
        # Record frames while the song is playing
        while pygame.mixer.music.get_busy():
            frames.append(stream.read(buffer))
            
        # Stop recording
        stream.stop_stream()
        stream.close()

        
        # Configure wave file settings
        wave_file = wave.open(new_file, 'wb')
        wave_file.setnchannels(channels)
        wave_file.setsampwidth(audio.get_sample_size(format))
        wave_file.setframerate(sample_rate)
        
        print("Saving " + new_file)   
        
        # Write the frames to the wave file
        wave_file.writeframes(b''.join(frames))
        wave_file.close()
        
        # Call FFmpeg to handle the MP3 conversion if desired
        if do_ffmpeg_convert:
            os.system('ffmpeg -i ' + new_file + ' -y -f mp3 -ab ' + str(mp3_bitrate) + 'k -ac ' + str(channels) + ' -ar ' + str(sample_rate) + ' -vn ' + file_name + '.mp3')
            
            # Delete the WAV file if desired
            if do_wav_cleanup:        
                os.remove(new_file)
        
    # End PyAudio    
    audio.terminate()    
 
except KeyboardInterrupt:
    # if user hits Ctrl/C then exit
    # (works only in console mode)
    pygame.mixer.music.fadeout(1000)
    pygame.mixer.music.stop()
    raise SystemExit