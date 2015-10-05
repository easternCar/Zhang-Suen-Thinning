
/***********************************************************************************
**	ZHANG SUEN thinning algorithm, for OpenCV										 **
**																				**
**															HammerAndSickle		**
************************************************************************************/

using namespace cv;

//@ Parameter Mat image should be binarized before do thinning (ex : by threshold() )
void thinning(Mat& image)
{
	image /= 255;		//white pixel : 255->1, black pixel : 0->0

	Mat prev = Mat::zeros(image.size(), CV_8UC1);		//Initially prev is to be prepared as zero matrix.
	Mat diff;											//Diffrence between before-thinned(prev) and after-thinned(image)

	//Go into loop until diff became zero matrix. (No diffrence were detected after thinning)
	do {
		thinningIteration(image, 1);		//step1
		thinningIteration(image, 2);		//step2

		//diff <- | image - prev |
		absdiff(image, prev, diff);

		//Save thinned image for this iteration. In next iteration this is to be referred to.
		image.copyTo(prev);

	} while (countNonZero(diff) > 0);
	//break when all values of diff are zero

	
	image *= 255;		//white pixel : 1->255, black pixel : 0->0
}


//@ Pixel(i, j) deletion is depends on 3 conditions.
// (1) Pixel(i,j)'s connectivity num == 1
// (2) Sum of pixels in every eight directions from Pixel(i,j) amounts to 2~8
// (3) Check some neighbor pixels from Pixel(i,j). Pixels to check depend on step
void thinningIteration(Mat& image, int step)
{
	//Make another Mat for marking rather than deleting image's pixels immediately
	Mat marker = Mat::zeros(image.size(), CV_8UC1);

	//Start double loop from (1, 1) to (row-1, col-1)
	for (int i = 1; i < image.rows - 1; i++)
	{
		for (int j = 1; j < image.cols - 1; j++)
		{
			int connectivityNum = 0;		//for condition (1)
			int neighborSum = 0;			//for condition (2)
			int neighborChecked[2] = { 0, 0 };	//for condition (3)

			uchar Npixels[8] = { image.at<uchar>(i, j + 1), image.at<uchar>(i + 1, j + 1), image.at<uchar>(i + 1, j), image.at<uchar>(i + 1, j - 1),
				image.at<uchar>(i, j - 1), image.at<uchar>(i - 1, j - 1), image.at<uchar>(i - 1, j), image.at<uchar>(i - 1, j + 1) };
			
			//{\A1\E6, \A2\D9, \A1\E9, \A2\D7, \A1\E7, \A2\D8, \A1\E8, \A2\D6}
			/*  Npixels[0] is a neighbor pixel to the right of the central Pixel(i,j)
				others are numbered in clockwise order.
			
				[5] [6]  [7]
				[4] [\A3\C0] [0]
				[3] [2]  [1]
			
			*/
			
			//====CONDITION (1)====//

			// Connectivity number of Pixel(i, j) = sum of Npixels[k]'s Connectivity number | k \A1\F4 set S
			// Connectivity num of Npixels[k] = Nipxels[k] - (Nipxels[k] * Nipxels[k+1] * Nipxels[k+2])			<indexing range : [0]~[7]> 
			
			// ex)	Npixels[6]'s connectivity : Npixels[6] - (Npixels[6] AND Npixels[7] AND Npixels[0])
			//		Npixels[2]'s connectivity : Npixels[2] - (Npixels[2] AND Npixels[3] AND Npixels[4])

			// set S is configurable. default : Npixels[0], Npixels[2], Npixels[4], Npixels[6]
			int set_S[8] = { 0, 2, 4, 6, -1, -1, -1, -1 };
			int Slen = 4;

			for (int Sidx = 0; Sidx < Slen; Sidx++)
				connectivityNum += (Npixels[set_S[Sidx]] & ~(Npixels[set_S[Sidx]] & Npixels[(set_S[Sidx] + 1) % 8] & Npixels[(set_S[Sidx] + 2) % 8]));

			//The for loop above as a default set is equal to :										
			/*connectivityNum = (Npixels[0] & ~(Npixels[0] & Npixels[1] & Npixels[2]))
				+ (Npixels[2] & ~(Npixels[2] & Npixels[3] & Npixels[4]))
				+ (Npixels[4] & ~(Npixels[4] & Npixels[5] & Npixels[6]))
				+ (Npixels[6] & ~(Npixels[6] & Npixels[7] & Npixels[0]));
				*/


			//====CONDITION (2)====//
			//Check sum of neighbor eight pixels
			for (int Nidx = 0; Nidx < 8; Nidx++)
				neighborSum += Npixels[Nidx];


			//====CONDITION (3)====//

			switch (step)
			{
			case 1: //step 1 : check for neighbor pixels (\A1\E7 AND \A1\E8 AND \A1\E6) and (\A1\E9 AND \A1\E7 AND \A1\E8)
				neighborChecked[0] = (Npixels[0] & Npixels[4] & Npixels[6]);
				neighborChecked[1] = (Npixels[2] & Npixels[4] & Npixels[6]);
				break;
			case 2: //step 2 : check for neighbor pixels (\A1\E6 AND \A1\E9 AND \A1\E8) and (\A1\E6 AND \A1\E9AND \A1\E7)
				neighborChecked[0] = (Npixels[0] & Npixels[2] & Npixels[6]);
				neighborChecked[1] = (Npixels[0] & Npixels[2] & Npixels[4]);
				break;
			}

			// Deletion condition
			// 1) connectivity == 1
			// 2) 2 <= sum of neighbors(num of white pixels) <= 6
			// 3) two results of neighbor pixels's AND operation == 1
			if ((connectivityNum == 1) && (neighborSum >= 2 && neighborSum <= 6) && (neighborChecked[0] == 0 && neighborChecked[1] == 0))
				marker.at<uchar>(i, j) = 1;		//Pixel(i,j) is to be deleted
		}
	}

	//After all loop through (m*n), delete every marked pixels from original image by (A AND ~B) operation
	image &= ~marker;
}


