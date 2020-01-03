# FlightEx
Experimental flight and ground data from Paparazzi UAS

- SciTech17 has the data and the code to generate the AIAA SciTech 17 conference paper graphs

- Flight_Data has some real flight data to be used for other projects


## Adding and pushing LFS files:

- First install git lfs if you havent done it already!

git lfs install

- then track the lfs files

git lfs track xxx.data

- add the file

git add xxx.data

- commit

git commit -m "Adding LFS data file"

- then push the lfs file

git lfs push origin master

- finally push the pointer to GitHub

git push origin master
 
