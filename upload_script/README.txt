To use this script:

SCRIPT SETUP

0.0) Download the upload_drive.sh file and place it in the folder your latest
participant's data files are currently located.
0.1) open the terminal and change directory to that folder (usually it will be
~/Downloads):

cd ~/Downloads

0.2) Give execution permission to the script by running the following command in
your terminal:

chmod +x upload_drive.sh

RUNNING THE SCRIPT

1.0) *read the "CONNECTING GOOGLE DRIVE" section below to learn to connect your Google Drive to your Terminal,
which is a prerequisite for this script to work.

!!!!!READ THIS NOT TO LOSE DATA!!!!!

PLEASE NOTE: to make sure your data is processed correctly, you have to run this
script for a folder that has only ONE participant's data. 

The script works by tagging all data files with a given participant's unique id
code, so it's REALLY important you either have deleted other participants' data from the current folder 
or created a separate data folder for every participant's individual data to Google Drive.

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

1.1) For every individual participant, you will run the following commands in your
terminal (changing "folder" to the filepath of the folder that has this particular
participant's data):

./upload_drive.sh folder

1.2) The script will sort the files into their correct folders automatically and
rename the files with a unique 10-digit participant id instead of a timestamp. 

You're done!

CONNECTING GOOGLE DRIVE

To establish a connection between your terminal and our research drive folder,
the script uses the rclone utility, which you need to install and configure to
work (only once).

2.0 installing rclone

simply execute this command in your terminal:

curl https://rclone.org/install.sh | sudo bash

2.1 making a remote connection to Google Drive

2.1.1 run the following command:

rclone config

When prompted, type "n" and press "Enter" to create a new remote.

2.1.2 name your remote

When prompted, enter the name of your remote as "crg" (use exactly this name!)
and press "Enter"

2.1.3 choose Google Drive connection type

When prompted for type of your remote connection drive, type "22" and press
"Enter".

2.1.4 client prompts (skip)
Press "Enter" when prompted for client id AND client secret.

2.1.5 access scope

When prompted for scope of your remote Google Drive access, type "1" and press
"Enter" for full access (needed for the script to work).

2.1.6 service account prompt (skip)
Press "enter" when prompted about service account.

2.1.7 advanced config

You will be prompted to enter advanced config, press "y" to do so.

2.1.7.0 Inside advanced config, keep pressing "Enter" until you are prompted for
"root_folder_id" (takes a few clicks)

2.1.7.1 When prompted for root folder id, copy and paste

1ERxyzAow04w8oQN0ClQZ_yOKMxRV8UeP

then press "Enter" (this is our DATA folder's unique address)

2.1.7.2 Keep pressing "Enter" until you are asked to "Enter advanced config?"
again, which is when you press "n" and "Enter" to exit.

2.1.8 web browser authentification
You will be asked "Use web browser to automatically authenticate rclone with
remote?", type "y" and press "Enter".

2.1.8 You should automatically be taken to a browser window where you will be
prompted to log into Google Drive, please do so.

After your remote is set up, you should be able to use the script as intended.
