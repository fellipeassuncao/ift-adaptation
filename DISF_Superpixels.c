/**
* Demo for the DISF algorithm
*
* @date October, 2019
*/

//=============================================================================
// Includes
//=============================================================================
#include "Image.h"
#include "DISF.h"
#include "Utils.h"

#include <time.h>
#include <stdio.h>

//=============================================================================
// Main
//=============================================================================
int main(int argc, char* argv[])
{
    int num_init_seeds, num_final_superpixels;
    Image *img, *border_img, *label_img, *ovlay_img;
    Graph *graph;
    clock_t time;

    // Validation of user's inputs
    if(argc != 4)
    {
        printf("Usage: disf [1] [2] [3]\n");
        printf("----------------------------------\n");
        printf("[1] - Input image (8-bit jpg, jpeg, png, ppm and pgm<P2>)\n" );
        printf("[2] - Initial number of seeds (e.g., 8000)\n");
        printf("[3] - Final number of superpixels\n");
        printError("main", "Too many/few parameters");
    }

    // Load image and get the user-defined params
    img = loadImage(argv[1]);
    num_init_seeds = atoi(argv[2]);
    num_final_superpixels = atoi(argv[3]);

    // Validate user inputs
    if(num_init_seeds <= 1)
        printError("main", "The number of initial seeds is too low");
    else if(num_final_superpixels <= 1)
        printError("main", "The number of final superpixels is too low");
    else if(num_init_seeds < num_final_superpixels)
        printError("main", "The number of initial seeds is lower than the desired quantity of superpixels");

    // Create auxiliary data structures
    border_img = createImage(img->num_rows, img->num_cols, 1);
    graph = createGraph(img);

    // // Run DISF algorithm
    time = clock();
    label_img = runDISF(img, num_init_seeds, num_final_superpixels, &border_img);
    time = clock() - time;

    // // Overlay superpixel's borders into a copy of the original image
    ovlay_img = overlayBorders(img, border_img, 1.0, 0.0, 0.0);

    printf("Time elapsed: %.3f seconds\n",((double)time)/CLOCKS_PER_SEC);

    // // Save the segmentation results
    writeImagePPM(ovlay_img, "ovlay.ppm");
    writeImagePGM(label_img, "labels.pgm");
    writeImagePGM(border_img, "borders.pgm");

    // // Free
    freeImage(&img);
    freeImage(&label_img);
    freeImage(&border_img);
    freeImage(&ovlay_img);
    freeGraph(&graph);
}