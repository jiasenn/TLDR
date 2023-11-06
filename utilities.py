def move(set_name):
    # path='../input/retinal-disease-classification/'+set_name+'_Set/'+set_name+'_Set/'
    path=set_name+'_Set/'+set_name+'_Set/'

    d_path=path+(set_name.split('_')[0] if set_name[0]!='E' else 'Validation')+'/'
    csv_path=path+'RFMiD_'
    if set_name[0]=='E':
        csv_path=csv_path+'Validation_Labels.csv'
    else:
        csv_path=csv_path+set_name+('ing' if set_name[1]=='e' else '')+'_Labels.csv'
    data=pd.read_csv(csv_path)
    # print(data)
    # print(d_path)
    for i in range(len(data.ID)):
        desired_data = 1
        row_index = i
        row = data.iloc[row_index][2:]  # Access the selected row
        if desired_data in row.values:
            column_name = row.index[row == desired_data][0]
        else:
            column_name = 'NORMAL'
        # print(column_name)
        image_name=str(i+1)+'.png'
        source_path=d_path+image_name
        destination_path='dataset/'+column_name+'/'+image_name
        shutil.copyfile(source_path, destination_path)

# Function to count images in a directory
def count_images(directory):
    image_count = len([file for file in os.listdir(directory) if file.lower().endswith(('.png', '.jpg', '.jpeg'))])
    return image_count

# Function to delete folders with fewer than 30 images
def delete_folders(path):
    for root, dirs, files in os.walk(path):
        for folder in dirs:
            folder_path = os.path.join(root, folder)
            image_count = count_images(folder_path)

            if image_count < 140:
                print(f"Folder: {folder_path} - Image count: {image_count}")
                try:
                    shutil.rmtree(folder_path)
                    print(f"Deleted folder: {folder}")
                except OSError as e:
                    print(f"Error: {e}")

def delete_extra_images(root_folder):
    subdir_images = {}
    min_images = float('inf')  # Set initial minimum count to infinity

    # Loop through the subdirectories to count the number of images in each
    for subdir in os.listdir(root_folder):
        subdir_path = os.path.join(root_folder, subdir)
        if os.path.isdir(subdir_path):
            subdir_images[subdir] = []
            for file in os.listdir(subdir_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    subdir_images[subdir].append(file)
            min_images = min(min_images, len(subdir_images[subdir]))

    # Delete extra images in each subdirectory
    for subdir, images in subdir_images.items():
        extra_images = len(images) - min_images
        if extra_images > 0:
            print(f"Deleting {extra_images} images in {subdir}")
            images_to_delete = random.sample(images, extra_images)  # Randomly select images for deletion
            for image in images_to_delete:
                os.remove(os.path.join(root_folder, subdir, image))

def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    ds_size = len(ds)

    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    # test_size = int(test_split * ds_size)

    train_ds=ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds=ds.skip(train_size).skip(val_size)

    return train_ds, val_ds, test_ds