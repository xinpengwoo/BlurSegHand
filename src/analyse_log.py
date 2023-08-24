import matplotlib.pyplot as plt

def plot_loss_trend(log_data, loss_name):
    epochs = []
    loss_values = []
    
    for entry in log_data:
        if loss_name in entry['loss_items']:
            loss_value = entry['loss_items'][loss_name]
            loss_values.append(loss_value)
            if entry['itr'] == 1:
                epochs.append(entry['epoch'])
            else:
                epochs.append('')
                
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(loss_values)), loss_values, marker='o')
    # when itrs[i]  1, show the order of epochs on xlabel 
    plt.xticks(range(len(loss_values)), epochs)

    plt.title(f'Variation of {loss_name} over Messages')
    plt.xlabel('Log Message (Epoch)')
    plt.ylabel(f'{loss_name} Value')
    plt.grid()
    plt.tight_layout()

    save_path = f'./{loss_name}_trend.png'
    plt.savefig(save_path)  # Save the plot to a file
    plt.close()  # Close the plot to free up memory


log_file_path = "/home/zzq/Xinpeng/BlurHand_RELEASE/experiments/BlurHandNet_BH/train_logs.txt" # Replace with the actual path to your log file

extracted_data = []

with open(log_file_path, 'r') as log_messages:
    for message in log_messages:
        message = message.split(' ', 2)[-1]  # Remove the timestamp
        if not message.startswith('Epoch'):  # Skip if the message isn't starting with 'Epoch'
            continue
        parts = message.split()
        epoch = int(parts[1].split('/')[0])
        itr = int(parts[3].split('/')[0])
        # remove colon from parts[3]
        parts[3] = parts[3][:-1]
        itr_per_epoch = int(parts[3].split('/')[1])
        lr = float(parts[5])
        speed = ' '.join(parts[7:10])
        
        loss_items = {}
        for i in range(10, len(parts), 2):
            if parts[i].startswith('loss_'):
                loss_name = parts[i][5:-1]
                loss_value = float(parts[i+1])
                loss_items[loss_name] = loss_value

        extracted_data.append({
            'epoch': epoch,
            'itr': itr,
            'itr_per_epoch': itr_per_epoch,
            'lr': lr,
            'speed': speed,
            'loss_items': loss_items
        })

# Print the extracted data for the first log message
print(extracted_data[-1])

plot_loss_trend(extracted_data, 'joint_cam')