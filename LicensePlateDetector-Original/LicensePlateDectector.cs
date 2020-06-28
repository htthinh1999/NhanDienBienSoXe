using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Text;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.OCR;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Emgu.Util;
using System.IO;

namespace LicensePlateDetector_Original
{
    public class LicensePlateDetector : DisposableObject
    {
        /// <summary>
        /// The OCR engine
        /// </summary>
        private Tesseract _ocr;

        public List<string> listWordsLongest = new List<string>();
        public Dictionary<string, IInputOutputArray> listPlateROIs = new Dictionary<string, IInputOutputArray>();
        public Dictionary<IInputOutputArray, IInputOutputArray> listPlateROIsColor = new Dictionary<IInputOutputArray, IInputOutputArray>();

        public IInputArray image;
        public IInputOutputArray plateROI = new Mat();
        public IInputOutputArray plateROI_Drawed = new Mat();

        /// <summary>
        /// Create a license plate detector
        /// </summary>
        /// <param name="dataPath">
        /// The datapath must be the name of the parent directory of tessdata and
        /// must end in / . Any name after the last / will be stripped.
        /// </param>
        public LicensePlateDetector(String dataPath)
        {
            //create OCR engine
            InitOcr(dataPath, "eng", OcrEngineMode.TesseractLstmCombined);

            _ocr.SetVariable("tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZ-.1234567890");
        }

        public static void TesseractDownloadLangFile(String folder, String lang)
        {
            String folderName = folder;
            if (!Directory.Exists(folderName))
            {
                Directory.CreateDirectory(folderName);
            }
            String dest = Path.Combine(folderName, String.Format("{0}.traineddata", lang));
            if (!File.Exists(dest))
                using (System.Net.WebClient webclient = new System.Net.WebClient())
                {
                    String source = Tesseract.GetLangFileUrl(lang);

                    Console.WriteLine(String.Format("Downloading file from '{0}' to '{1}'", source, dest));
                    webclient.DownloadFile(source, dest);
                    Console.WriteLine(String.Format("Download completed"));
                }
        }

        private void InitOcr(String path, String lang, OcrEngineMode mode)
        {
            try
            {
                if (_ocr != null)
                {
                    _ocr.Dispose();
                    _ocr = null;
                }

                if (String.IsNullOrEmpty(path))
                    path = Tesseract.DefaultTesseractDirectory;

                TesseractDownloadLangFile(path, lang);
                TesseractDownloadLangFile(path, "osd"); //script orientation detection

                _ocr = new Tesseract(path, lang, mode);
            }
            catch (System.Net.WebException e)
            {
                _ocr = null;
                throw new Exception("Unable to download tesseract lang file. Please check internet connection.", e);
            }
            catch (Exception e)
            {
                _ocr = null;
                throw e;
            }
        }



        /*
        /// <summary>
        /// Compute the white pixel mask for the given image. 
        /// A white pixel is a pixel where:  saturation &lt; 40 AND value &gt; 200
        /// </summary>
        /// <param name="image">The color image to find white mask from</param>
        /// <returns>The white pixel mask</returns>
        private static Image<Gray, Byte> GetWhitePixelMask(Image<Bgr, byte> image)
        {
           using (Image<Hsv, Byte> hsv = image.Convert<Hsv, Byte>())
           {
              Image<Gray, Byte>[] channels = hsv.Split();

              try
              {
                 //channels[1] is the mask for satuation less than 40, this is the mask for either white or black pixels
                 channels[1]._ThresholdBinaryInv(new Gray(40), new Gray(255));

                 //channels[2] is the mask for bright pixels
                 channels[2]._ThresholdBinary(new Gray(200), new Gray(255));

                 CvInvoke.BitwiseAnd(channels[1], channels[2], channels[0], null);
              }
              finally
              {
                 channels[1].Dispose();
                 channels[2].Dispose();
              }
              return channels[0];
           }
        }*/

        /// <summary>
        /// Detect license plate from the given image
        /// </summary>
        /// <param name="img">The image to search license plate from</param>
        /// <param name="licensePlateImagesList">A list of images where the detected license plate regions are stored</param>
        /// <param name="filteredLicensePlateImagesList">A list of images where the detected license plate regions (with noise removed) are stored</param>
        /// <param name="detectedLicensePlateRegionList">A list where the regions of license plate (defined by an MCvBox2D) are stored</param>
        /// <returns>The list of words for each license plate</returns>
        public List<String> DetectLicensePlate(
           IInputArray img,
           List<IInputOutputArray> licensePlateImagesList,
           List<IInputOutputArray> filteredLicensePlateImagesList,
           List<RotatedRect> detectedLicensePlateRegionList)
        {
            image = img;
            List<String> licenses = new List<String>();
            using (Mat gray = new Mat())
            using (Mat canny = new Mat())
            using (VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint())
            {
                CvInvoke.CvtColor(img, gray, ColorConversion.Bgr2Gray);
                CvInvoke.Canny(gray, canny, 100, 50, 3, false);
                int[,] hierachy = CvInvoke.FindContourTree(canny, contours, ChainApproxMethod.ChainApproxSimple);

                FindLicensePlate(contours, hierachy, 0, gray, canny, licensePlateImagesList, filteredLicensePlateImagesList, detectedLicensePlateRegionList, licenses);
            }
            return licenses;
        }

        private static int GetNumberOfChildren(int[,] hierachy, int idx)
        {
            //first child
            idx = hierachy[idx, 2];
            if (idx < 0)
                return 0;

            int count = 1;
            while (hierachy[idx, 0] > 0)
            {
                count++;
                idx = hierachy[idx, 0];
            }
            return count;
        }

        private void FindLicensePlate(
           VectorOfVectorOfPoint contours, int[,] hierachy, int idx, IInputArray gray, IInputArray canny,
           List<IInputOutputArray> licensePlateImagesList, List<IInputOutputArray> filteredLicensePlateImagesList, List<RotatedRect> detectedLicensePlateRegionList,
           List<String> licenses)
        {
            for (; idx >= 0; idx = hierachy[idx, 0])
            {
                int numberOfChildren = GetNumberOfChildren(hierachy, idx);
                //if it does not contains any children (charactor), it is not a license plate region
                if (numberOfChildren == 0) continue;

                using (VectorOfPoint contour = contours[idx])
                {
                    if (CvInvoke.ContourArea(contour) > 400)
                    {
                        if (numberOfChildren < 3)
                        {
                            //If the contour has less than 3 children, it is not a license plate (assuming license plate has at least 3 charactor)
                            //However we should search the children of this contour to see if any of them is a license plate
                            FindLicensePlate(contours, hierachy, hierachy[idx, 2], gray, canny, licensePlateImagesList,
                               filteredLicensePlateImagesList, detectedLicensePlateRegionList, licenses);
                            continue;
                        }

                        RotatedRect box = CvInvoke.MinAreaRect(contour);
                        if (box.Angle < -45.0)
                        {
                            float tmp = box.Size.Width;
                            box.Size.Width = box.Size.Height;
                            box.Size.Height = tmp;
                            box.Angle += 90.0f;
                        }
                        else if (box.Angle > 45.0)
                        {
                            float tmp = box.Size.Width;
                            box.Size.Width = box.Size.Height;
                            box.Size.Height = tmp;
                            box.Angle -= 90.0f;
                        }

                        double whRatio = (double)box.Size.Width / box.Size.Height;
                        if (!(3.0 < whRatio && whRatio < 10.0))
                        //if (!(1.0 < whRatio && whRatio < 2.0))
                        {
                            //if the width height ratio is not in the specific range,it is not a license plate 
                            //However we should search the children of this contour to see if any of them is a license plate
                            //Contour<Point> child = contours.VNext;
                            if (hierachy[idx, 2] > 0)
                                FindLicensePlate(contours, hierachy, hierachy[idx, 2], gray, canny, licensePlateImagesList,
                                   filteredLicensePlateImagesList, detectedLicensePlateRegionList, licenses);
                            continue;
                        }
                        using (UMat tmpPlate1 = new UMat())
                        using (UMat tmpPlate2 = new UMat())
                        using (UMat tmpPlateROI1 = new UMat())
                        using (UMat tmpPlateROI2 = new UMat())
                        using (UMat tmp1 = new UMat())
                        using (UMat tmp2 = new UMat())
                        {
                            PointF[] srcCorners = box.GetVertices();

                            PointF[] destCorners = new PointF[] {
                                new PointF(0, box.Size.Height - 1),
                                new PointF(0, 0),
                                new PointF(box.Size.Width - 1, 0),
                                new PointF(box.Size.Width - 1, box.Size.Height - 1)};

                            using (Mat rot = CvInvoke.GetAffineTransform(srcCorners, destCorners))
                            {
                                CvInvoke.WarpAffine(gray, tmp1, rot, Size.Round(box.Size));
                                // Get Color Plate ROI
                                CvInvoke.WarpAffine(image, tmpPlate1, rot, Size.Round(box.Size));
                                CvInvoke.WarpAffine(image, tmpPlateROI1, rot, Size.Round(box.Size));

                            }
                            //resize the license plate such that the front is ~ 10-12. This size of front results in better accuracy from tesseract
                            Size approxSize = new Size(240, 180);
                            double scale = Math.Min(approxSize.Width / box.Size.Width, approxSize.Height / box.Size.Height);
                            Size newSize = new Size((int)Math.Round(box.Size.Width * scale), (int)Math.Round(box.Size.Height * scale));
                            CvInvoke.Resize(tmp1, tmp2, newSize, 0, 0, Inter.Cubic);
                            CvInvoke.Resize(tmpPlate1, tmpPlate2, newSize, 0, 0, Inter.Cubic);
                            CvInvoke.Resize(tmpPlateROI1, tmpPlateROI2, newSize, 0, 0, Inter.Cubic);
                            //CvInvoke.Imshow("tmpPlate2", tmpPlate2);

                            //removes some pixels from the edge
                            int edgePixelSize = 3;
                            Rectangle newRoi = new Rectangle(new Point(edgePixelSize, edgePixelSize),
                               tmp2.Size - new Size(2 * edgePixelSize, 2 * edgePixelSize));
                            UMat plate = new UMat(tmp2, newRoi);
                            UMat plateTemp = new UMat(tmpPlate2, newRoi);
                            UMat plateROITemp = new UMat(tmpPlateROI2, newRoi);
                            plateROI = plateROITemp;
                            plateROI_Drawed = plateTemp;
                            //CvInvoke.Imshow("plateROI" + idx, plateROI);

                            UMat filteredPlate = FilterPlate(plate);
                            //Tesseract.Character[] words;
                            StringBuilder strBuilder = new StringBuilder();
                            using (UMat tmp = filteredPlate.Clone())
                            {
                                //CvInvoke.Imshow("tmp" + idx, tmp);
                                _ocr.SetImage(tmp);
                                _ocr.Recognize();

                                strBuilder.Append(_ocr.GetUTF8Text());
                                /*
                                words = _ocr.GetCharacters();

                                if (words.Length == 0) continue;

                                for (int i = 0; i < words.Length; i++)
                                {
                                    strBuilder.Append(words[i].Text);
                                }*/
                            }

                            licenses.Add(strBuilder.ToString());
                            licensePlateImagesList.Add(plate);
                            filteredLicensePlateImagesList.Add(filteredPlate);
                            detectedLicensePlateRegionList.Add(box);
                            if ((listWordsLongest.Count == 0 || listWordsLongest[listWordsLongest.Count - 1].Length < strBuilder.ToString().Length))
                            {
                                listWordsLongest.Add(strBuilder.ToString());
                                listPlateROIs.Add(strBuilder.ToString(), plateROI);
                                listPlateROIsColor.Add(plate, plateROI_Drawed);
                            }
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Filter the license plate to remove noise
        /// </summary>
        /// <param name="plate">The license plate image</param>
        /// <returns>License plate image without the noise</returns>
        private static UMat FilterPlate(UMat plate)
        {
            UMat thresh = new UMat();
            CvInvoke.Threshold(plate, thresh, 120, 255, ThresholdType.BinaryInv);
            //Image<Gray, Byte> thresh = plate.ThresholdBinaryInv(new Gray(120), new Gray(255));

            Size plateSize = plate.Size;
            using (Mat plateMask = new Mat(plateSize.Height, plateSize.Width, DepthType.Cv8U, 1))
            using (Mat plateCanny = new Mat())
            using (VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint())
            {
                plateMask.SetTo(new MCvScalar(255.0));
                CvInvoke.Canny(plate, plateCanny, 100, 50);
                CvInvoke.FindContours(plateCanny, contours, null, RetrType.External, ChainApproxMethod.ChainApproxSimple);

                int count = contours.Size;
                for (int i = 0; i < count; i++)
                {
                    using (VectorOfPoint contour = contours[i])
                    {
                        Rectangle rect = CvInvoke.BoundingRectangle(contour);
                        if (rect.Height > (plateSize.Height >> 1))
                        {
                            rect.X -= 1; rect.Y -= 1; rect.Width += 2; rect.Height += 2;
                            Rectangle roi = new Rectangle(Point.Empty, plate.Size);
                            rect.Intersect(roi);
                            CvInvoke.Rectangle(plateMask, rect, new MCvScalar(), -1);
                            //plateMask.Draw(rect, new Gray(0.0), -1);
                        }
                    }

                }

                thresh.SetTo(new MCvScalar(), plateMask);
            }

            CvInvoke.Erode(thresh, thresh, null, new Point(-1, -1), 1, BorderType.Constant, CvInvoke.MorphologyDefaultBorderValue);
            CvInvoke.Dilate(thresh, thresh, null, new Point(-1, -1), 1, BorderType.Constant, CvInvoke.MorphologyDefaultBorderValue);

            return thresh;
        }

        protected override void DisposeObject()
        {
            _ocr.Dispose();
        }

        public IInputOutputArray Segmentation(IInputOutputArray image)
        {
            IInputOutputArray plate_cut = image;
   
            //CvInvoke.Imshow("test6", plate_cut);
            IOutputArray plate_blur = new Mat();
            CvInvoke.GaussianBlur(plate_cut, plate_blur, new Size(3, 3), 1); //làm mịn ảnh
            //CvInvoke.Imshow("test7", plate_blur);
            IOutputArray plate_thres = new Mat();
            CvInvoke.Threshold(plate_blur, plate_thres, 120, 255, Emgu.CV.CvEnum.ThresholdType.BinaryInv); //phân ngưỡng ảnh
            //CvInvoke.Imshow("test8", plate_thres);
            IInputArray kernel2 = CvInvoke.GetStructuringElement(Emgu.CV.CvEnum.ElementShape.Rectangle, new Size(3, 3), new Point(-1, -1));
            IInputOutputArray plate_thres_mor = new Mat();
            CvInvoke.MorphologyEx(plate_thres, plate_thres_mor, Emgu.CV.CvEnum.MorphOp.Dilate, kernel2, new Point(-1, -1), 0, Emgu.CV.CvEnum.BorderType.Constant, CvInvoke.MorphologyDefaultBorderValue); //giảm edge nhiễu làm sắc các edge bằng kerel(3,3)
            //CvInvoke.Imshow("test9", plate_thres_mor);
            VectorOfVectorOfPoint plate_contour = new VectorOfVectorOfPoint();
            IOutputArray plate_hier = new Mat();
            CvInvoke.FindContours(plate_thres_mor, plate_contour, plate_hier, Emgu.CV.CvEnum.RetrType.External, Emgu.CV.CvEnum.ChainApproxMethod.ChainApproxSimple); //tìm contour chữ và số
            
            IInputOutputArray plate_draw = listPlateROIsColor[image]; // ảnh gốc biển số bị cắt rời
         
            //CvInvoke.Imshow("test10", plate_draw);
            // Khoanh vung chu
            Dictionary<double, int> areas_ind = new Dictionary<double, int>(); // lưu diện tích contour và vị trí contour
            List<double> areas = new List<double>();
            for (int i = 0; i < plate_contour.Size; i++)
            {
                double area = CvInvoke.ContourArea(plate_contour[i]);
                areas_ind[area] = i; // lưu vị trí 
                areas.Add(area); // thêm diện tích contour  
            }
            areas.Sort((a, b) => b.CompareTo(a)); // sắp xếp diện tích các contour từ lớn đến nhỏ
            List<double> areas1 = new List<double>();
            for (int i = 0; i < Math.Min(10, areas.Count); i++)
            {
                areas1.Add(areas[i]); // thêm diện tích 10 contour đầu tiên sau khi sắp xếp
            }
            for (int i = 0; i < areas1.Count; i++)
            {
                Rectangle rectangle = CvInvoke.BoundingRectangle(plate_contour[areas_ind[areas1[i]]]); // khung bao quanh kí tự
                CvInvoke.Rectangle(plate_draw,
                    //new Rectangle(rectangle.X, rectangle.Y, rectangle.X + rectangle.Width, rectangle.Y + rectangle.Height),
                    rectangle,
                    new MCvScalar(0, 255, 0), 2); 
            }
            //CvInvoke.Imshow("Segmentation", plate_draw);
            return plate_draw;
            //CvInvoke.Imshow("Segmentation", _licensePlateDetector.plateROI);
        }
    }
}
