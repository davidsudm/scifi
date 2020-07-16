
import numpy as np
import matplotlib.pyplot as plt
from drs4 import DRS4BinaryFile

with DRS4BinaryFile('/Users/lonewolf/Desktop/rootfiles/mu3e_data/scsf78-30cm.dat') as f:

    board_id = f.board_ids[0]
    active_channels = f.channels
    print('DRS4 board id: {}'.format(board_id))
    print('Active channels: {}'.format(active_channels[board_id]))

    time_widths = [f.time_widths[board_id][1],
                   f.time_widths[board_id][2],
                   f.time_widths[board_id][3],
                   f.time_widths[board_id][4]]

    times = np.array([np.cumsum(time_widths[0]),
                      np.cumsum(time_widths[1]),
                      np.cumsum(time_widths[2]),
                      np.cumsum(time_widths[3])])

    # times = np.insert(times, 0, 0, axis=1)

    #event = next(f)

    for event in f:

        event_id = event.event_id
        event_timestamp = event.timestamp
        event_range_center = event.range_center
        event_scalers = [event.scalers[board_id][1],
                         event.scalers[board_id][2],
                         event.scalers[board_id][3],
                         event.scalers[board_id][4]]
        event_trigger_cell = event.trigger_cells[board_id]

        print('Event id : {}'.format(event_id))
        print('Event timestamp : {}'.format(event_timestamp))
        print('Event range center : {}'.format(event_range_center))
        print('Event scalers : {}'.format(event_scalers))
        print('Event trigger cell : {}'.format(event_trigger_cell))

        adc_data = [event.adc_data[board_id][1],
                    event.adc_data[board_id][2],
                    event.adc_data[board_id][3],
                    event.adc_data[board_id][4]]

        adc_data = np.insert(adc_data, 0, 0, axis=1)

        fig, axs = plt.subplots(2, 2, sharex="True", sharey="True")

        # Triggers
        axs[0, 0].plot(times[1], adc_data[1], color='red')
        axs[1, 0].plot(times[3], adc_data[3], color='red')
        # Signals
        axs[0, 1].plot(times[0], adc_data[0])
        axs[1, 1].plot(times[2], adc_data[2])

        axs[1, 0].set_xlabel(r'$Time_{signal}$ [ns]')
        axs[1, 1].set_xlabel(r'$Time_{trigger}$ [ns]')
        axs[0, 0].set_ylabel('Integer voltage')
        axs[1, 0].set_ylabel('Integer voltage')

        plt.show()
