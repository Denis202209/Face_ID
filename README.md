Detailed instructions on how to run facial recognition software on windows operating system:

<b>Step 1: Install the required software</b>
<br>
<br>
PyCharm Installation:
Go to the PyCharm Community Edition link and download the installer (https://pycharm-community-edition.softonic.ru/).
<br>
Follow the instructions to install it on your computer.
<br>
Installing MySQL:
<br>
Download MySQL Installer from the official website(https://dev.mysql.com/downloads/installer/).
<br>
Perform the installation by selecting the required components including MySQL Server and MySQL Workbench.
<br>
<br>
<b>Step 2: Configure MySQL</b>
<br>
<br>
1.Navigate to the MySQL Server directory:
<br>
Open a Windows command prompt and run the command:
<br>
*cd "C:\Program Files\MySQL\MySQL Server 8.3" && cd .\bin*
<br>
2.Start the MySQL Client:
<br>
Enter the following command to start the client:
<br>
*.\mysql -u root -p*
<br>
 Enter your password.
 <br>
3.Create Database:
<br>
In the MySQL command interface, execute:
<br>
*create database Authorized_user;*
<br>
4.Selecting the database to use:
<br>
Execute the command:
<br>
*use Authorized_user;*
<br>
Create a table in the database. Enter the SQL command to create the table:
<br>
*CREATE TABLE my_table (id int, Name varchar(255), Age varchar(255), Address varchar(255));*
<br>
<br>
<b>Step 3: Install Python dependencies</b>
<br>
<br>
Open PyCharm terminal or any other command interface and install the following packages using pip:
<br>
*pip install opencv-python==4.3.0.38
pip install mtcnn
pip install keras_preprocessing
pip install Pillow
pip install h5py
pip install mysql-connector-python
pip install scikit-learn
pip install tensorflow*
<br>
<br>
<b>Step 4: Setting up a project in PyCharm</b>
<br>
<br>
Create a new project in PyCharm:
<br>
Open PyCharm and select File > New Project.
<br>
Specify the path to the project directory and the Python interpreter to be used.
<br>
Adding project files. Place the following files in the project directory:
<br>
__init__.py
load_dataset.py
gui.py
train_classifier.py
vgg16_train.py
img_navigator.py
haarcascade_frontalface_default.xml
<br>
<br>
<b>Step 5: Run the program (start the GUI):</b>
<br>
<br>
in PyCharm, open the gui.py file and run the script by Right-clicking > Run 'gui'.
<br>
Once all these steps are completed, the face recognition program will be ready to use. You will be able to enter data, train the classifier, recognize faces, and manage violation data through the graphical user interface.
