import os
import time
import csv
from copy import deepcopy, copy
class CSV_Logger:
    def __init__(self, fieldnames, args, iteration_fieldnames=['epoch', 'episode', 'step'], recover_filename=None):
        if recover_filename is not None:
            self.csv_filename = recover_filename
        else:
            self.csv_file_path = '/home/erick/log_leap/pnr/' + time.strftime('%Y_%m_%d_%H_%M_%S') +\
                                 'train_tdm_loss_log.csv'
        self.fieldnames = fieldnames
        self.iteration_fieldnames = iteration_fieldnames
        all_fieldnames = self.iteration_fieldnames + self.fieldnames
        if (not os.path.isfile(self.csv_file_path)) or os.stat(self.csv_file_path).st_size == 0:
            with open(self.csv_file_path, 'w') as csv_file:
                self.writer = csv.DictWriter(csv_file, fieldnames=all_fieldnames)
                self.writer.writeheader()

        self.entries = {}
        self.num_entries = {}
        for k in self.fieldnames + self.iteration_fieldnames:
            self.entries[k] = []
            self.num_entries[k] = 0

    def add_log(self, keyname, val):
        if keyname not in self.fieldnames:
            raise Exception("The given keyname does not belong to the valid keynames for this logger")
        else:
            self.entries[keyname].append(copy(val))
            self.num_entries[keyname] += 1

    def finish_step_log(self, step):
        self._finish_it_log('step', step)

    def finish_episode_log(self, episode):
        self._finish_it_log('episode', episode)

    def finish_epoch_log(self, epoch):
        self._finish_it_log('epoch', epoch)
        self.write_to_csv_file()
        for k in self.fieldnames + self.iteration_fieldnames:
            self.entries[k] = []
            self.num_entries[k] = 0

    def _finish_it_log(self, it_keyname, it):
        # we see ehich from the entries has the maximum amount
        max_amount = -1
        for k in self.fieldnames:
            if self.num_entries[k] > max_amount:
                max_amount = self.num_entries[k]
        # the amount of logged value until this call must belong to this iteration; therefore each must have this is
        # we calculate how many
        it_diff = max_amount - self.num_entries[it_keyname]
        assert it_diff >= 0
        it_index = self.iteration_fieldnames.index(it_keyname)
        if it_index == len(self.iteration_fieldnames) - 1:
            prev_diff = it_diff
        else:
            prev_diff = copy(max_amount - self.num_entries[self.iteration_fieldnames[it_index + 1]])
        if it_diff == 0:
            # No new entries
            self.entries[it_keyname].append(None)
            self.num_entries[it_keyname] += 1
            for k in self.fieldnames + self.iteration_fieldnames[it_index + 1:]:
                self.entries[k].append(None)
                self.num_entries[k] += 1
        else:
            for _ in range(it_diff):
                self.entries[it_keyname].append(it)
            prev_it_amount = deepcopy(self.num_entries[it_keyname])
            self.num_entries[it_keyname] = max_amount
            # Now we are going to pad for those values that were not logged as often; should not be too much. But for example
            # we could get a loss for each iteration inside an episode and one reward per episode

            for k in self.fieldnames + self.iteration_fieldnames[it_index + 1:]:
                diff = max_amount - self.num_entries[k]
                assert diff >= 0 and diff <= it_diff
                if diff == 0:
                    continue
                assert prev_diff <= diff
                pad_after = [None] * prev_diff
                pad = [None] * (diff - prev_diff)
                new_entries = self.num_entries[k] - prev_it_amount
                if new_entries > 0:
                    # we move the information to the last part
                    self.entries[k] = self.entries[k][:-new_entries] + pad + self.entries[k][-new_entries:] + pad_after
                else:
                    # There is no new entry and we just fill with None
                    self.entries[k] += pad + pad_after
                self.num_entries[k] = max_amount

    def write_to_csv_file(self):
        all_fieldnames = self.iteration_fieldnames + self.fieldnames
        with open(self.csv_file_path, 'a') as csv_file:
            self.writer = csv.DictWriter(csv_file, fieldnames=all_fieldnames)
            for i in range(len(self.entries['epoch'])):
                row = {}
                for k in all_fieldnames:
                    row[k] = self.entries[k][i]
                self.writer.writerow(row)