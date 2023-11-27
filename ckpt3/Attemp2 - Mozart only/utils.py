import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data

# from midox import midiread, midiwrite
import pretty_midi
import numpy as np
from sklearn.model_selection import train_test_split
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIDI_PIANO_PROGRAMS = 8

def parse_piano(file_path):
    try:
        # print("Parsing piano from file:", file_path)
        midi_data = pretty_midi.PrettyMIDI(file_path)
    except:
        # print("Cannot open midi file:", file_path)
        return None
    
    if midi_data==None:
        raise Exception("midi_data is None")

    instruments_to_remove = []

    for instrument in midi_data.instruments:
        if instrument.program > MIDI_PIANO_PROGRAMS or instrument.is_drum:
            instruments_to_remove.append(instrument)

    for i in instruments_to_remove:
        midi_data.instruments.remove(i)

    return midi_data

class NotesGenerationDataset(Data.Dataset):

    def __init__(self, midi_folder_path, longest_sequence_length=None, adl_dataset=False, type='train'):
        
        # self.midi_folder_path = midi_folder_path
        self.longest_sequence_length = longest_sequence_length

        if adl_dataset:
            assert type in ['train', 'val', 'test']
            self.midi_full_filenames = []

            for foldername, subfolders, filenames in os.walk(midi_folder_path):
                for filename in filenames:
                    file_path = os.path.join(foldername, filename)
                    if parse_piano(file_path) is not None:
                        self.midi_full_filenames.append(file_path)

            # Splitting the filenames into train, val, and test sets
            train_filenames, test_filenames = train_test_split(self.midi_full_filenames, test_size=0.3, random_state=42)
            val_filenames, test_filenames = train_test_split(test_filenames, test_size=0.5, random_state=42)

            if type=='train':
                self.midi_full_filenames = train_filenames
            elif type=='val':
                self.midi_full_filenames = val_filenames
            else:
                self.midi_full_filenames = test_filenames
        else:
            midi_filenames = os.listdir(midi_folder_path)
            self.midi_full_filenames = list(map(lambda filename: os.path.join(midi_folder_path, filename), midi_filenames))
        
        if longest_sequence_length is None:
            self.update_the_max_length()

    def midi_filename_to_piano_roll(self, midi_filename):
        # # midi_data = midiread(midi_filename, dt=0.3)
        # piano_roll = midi_data.piano_roll.transpose()
        midi_data = parse_piano(midi_filename)
        if midi_data is None:
            self.midi_full_filenames.remove(midi_filename)
            return None

        piano_roll = midi_data.get_piano_roll()

        # Pressed notes are replaced by 1
        piano_roll[piano_roll > 0] = 1
        
        return piano_roll

    def pad_piano_roll(self, piano_roll, max_length=132333, pad_value=0):
        original_piano_roll_length = piano_roll.shape[1]
        
        padded_piano_roll = np.zeros((piano_roll.shape[0], max_length))
        padded_piano_roll[:] = pad_value
        
        padded_piano_roll[:, -original_piano_roll_length:] = piano_roll

        return padded_piano_roll
    
    
    def update_the_max_length(self):
        sequences_lengths = []
        for filename in self.midi_full_filenames:
            piano_roll = self.midi_filename_to_piano_roll(filename)
            if piano_roll is not None:
                sequences_lengths.append(piano_roll.shape[1])
        if sequences_lengths:
            max_length = max(sequences_lengths)
            self.longest_sequence_length = max_length
                
    
    def __len__(self):
        return len(self.midi_full_filenames)
    
    def __getitem__(self, index):
        midi_full_filename = self.midi_full_filenames[index]
        piano_roll = self.midi_filename_to_piano_roll(midi_full_filename)
        
        # Shifting by one time step
        sequence_length = piano_roll.shape[1] - 1
        
        # Shifting by one time step
        input_sequence = piano_roll[:, :-1]
        ground_truth_sequence = piano_roll[:, 1:]
                
        # padding sequence so that all of them have the same length
        input_sequence_padded = self.pad_piano_roll(input_sequence, max_length=self.longest_sequence_length)
        
        ground_truth_sequence_padded = self.pad_piano_roll(ground_truth_sequence,max_length=self.longest_sequence_length,pad_value=-100)
                
        input_sequence_padded = input_sequence_padded.transpose()
        ground_truth_sequence_padded = ground_truth_sequence_padded.transpose()
        
        return (torch.FloatTensor(input_sequence_padded),torch.LongTensor(ground_truth_sequence_padded),torch.LongTensor([sequence_length]) )


def post_process_sequence_batch(batch_tuple):
    input_sequences, output_sequences, lengths = batch_tuple
    
    splitted_input_sequence_batch = input_sequences.split(split_size=1)
    splitted_output_sequence_batch = output_sequences.split(split_size=1)
    splitted_lengths_batch = lengths.split(split_size=1)

    training_data_tuples = zip(splitted_input_sequence_batch,
                               splitted_output_sequence_batch,
                               splitted_lengths_batch)

    training_data_tuples_sorted = sorted(training_data_tuples,
                                         key=lambda p: int(p[2]),
                                         reverse=True)

    splitted_input_sequence_batch, splitted_output_sequence_batch, splitted_lengths_batch = zip(*training_data_tuples_sorted)

    input_sequence_batch_sorted = torch.cat(splitted_input_sequence_batch)
    output_sequence_batch_sorted = torch.cat(splitted_output_sequence_batch)
    lengths_batch_sorted = torch.cat(splitted_lengths_batch)
    
    input_sequence_batch_sorted = input_sequence_batch_sorted[:, -lengths_batch_sorted[0, 0]:, :]
    output_sequence_batch_sorted = output_sequence_batch_sorted[:, -lengths_batch_sorted[0, 0]:, :]
    
    input_sequence_batch_transposed = input_sequence_batch_sorted.transpose(0, 1)
    
    lengths_batch_sorted_list = list(lengths_batch_sorted)
    lengths_batch_sorted_list = map(lambda x: int(x), lengths_batch_sorted_list)
    
    return input_sequence_batch_transposed, output_sequence_batch_sorted, list(lengths_batch_sorted_list)

def validate(model, valset_loader, criterion_val):
    model.eval()
    full_val_loss = 0.0
    overall_sequence_length = 0.0
    keys_shape = 88

    with torch.no_grad():
        for batch in tqdm(valset_loader):
            post_processed_batch_tuple = post_process_sequence_batch(batch)

            input_sequences_batch, output_sequences_batch, sequences_lengths = post_processed_batch_tuple
            output_sequences_batch_var =  Variable(output_sequences_batch.contiguous().view(-1).to(device))
            input_sequences_batch_var = Variable(input_sequences_batch.to(device))

            logits, _ = model(input_sequences_batch_var, sequences_lengths)

            loss = criterion_val(logits, output_sequences_batch_var)

            full_val_loss += loss.item()
            overall_sequence_length += sum(sequences_lengths)
            keys_shape = input_sequences_batch.shape[2]
    
    return full_val_loss / (overall_sequence_length * keys_shape)

def train_model(model, lrs_triangular, trainset_loader, criterion, valset_loader, criterion_val, epochs_number=2, wd=0.0, best_val_loss=float("inf"), clip=1.0, save=True):
    loss_list = []
    val_list =[]
    optimizer = torch.optim.Adam(model.parameters(), lr=lrs_triangular[0], weight_decay=wd)
    for epoch_number in range(epochs_number):
        model.train()
        epoch_loss = []
        print("Epoch: ", epoch_number)
        for lr, batch in tqdm(zip(lrs_triangular, trainset_loader)):
            optimizer.param_groups[0]['lr'] = lr

            post_processed_batch_tuple = post_process_sequence_batch(batch)
            input_sequences_batch, output_sequences_batch, sequences_lengths = post_processed_batch_tuple
            output_sequences_batch_var =  Variable(output_sequences_batch.contiguous().view(-1).to(device))
            input_sequences_batch_var = Variable(input_sequences_batch.to(device))

            optimizer.zero_grad()

            logits, _ = model(input_sequences_batch_var, sequences_lengths)

            loss = criterion(logits, output_sequences_batch_var)
            loss_list.append(loss.item())
            epoch_loss.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        current_trn_epoch = sum(epoch_loss)/len(trainset_loader)
        print('Training Loss: Epoch:',epoch_number,':', current_trn_epoch)

        current_val_loss = validate(model, valset_loader, criterion_val)
        print('Validation Loss: Epoch:',epoch_number,':', current_val_loss)
        print('')

        val_list.append(current_val_loss)

        if current_val_loss < best_val_loss:
            if save:
                torch.save(model.state_dict(), 'music_model_padfront_regularized.pth')
            best_val_loss = current_val_loss
    
    return best_val_loss