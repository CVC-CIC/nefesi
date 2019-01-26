from nefesi.util.segmentation.Broden_analize import Segment_images



def main():
    matrixfinal= Segment_images(['/data/114-1/datasets/unifiedparsing/ILSVRC2012_val_00035623.JPEG','/data/114-1/datasets/unifiedparsing/ILSVRC2012_val_00009396.JPEG'])
    print((matrixfinal[0]['object']).shape)

if __name__ == '__main__':
    main()

