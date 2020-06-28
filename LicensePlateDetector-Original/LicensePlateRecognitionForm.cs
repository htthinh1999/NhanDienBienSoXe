using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.UI;
using System.Diagnostics;
using Emgu.CV.Util;
using System.Security.Cryptography;

namespace LicensePlateDetector_Original
{
    public partial class LicensePlateRecognitionForm : Form
    {
        private LicensePlateDetector _licensePlateDetector;

        public LicensePlateRecognitionForm()
        {
            InitializeComponent();

            //System.Net.ServicePointManager.Expect100Continue = true;
            //System.Net.ServicePointManager.SecurityProtocol = System.Net.SecurityProtocolType.Tls12;

            _licensePlateDetector = new LicensePlateDetector("");
            Mat m = new Mat("license-plate.jpg");
            UMat um = m.GetUMat(AccessType.ReadWrite);
            imageBox1.Image = um;
            txtFileDirectory.Text = "license-plate.jpg";
            ProcessImage(m);
        }

        private void ProcessImage(IInputOutputArray image)
        {
            Stopwatch watch = Stopwatch.StartNew(); // Đếm thời gian
            List<IInputOutputArray> licensePlateImagesList = new List<IInputOutputArray>();//danh sách các ảnh có thể là biển số(hcn)
            List<IInputOutputArray> filteredLicensePlateImagesList = new List<IInputOutputArray>();// danh sách các ảnh đã được lọc nhiễu
            List<RotatedRect> licenseBoxList = new List<RotatedRect>();//Vị trí các bức ảnh  
            List<string> words = _licensePlateDetector.DetectLicensePlate( //Tìm kiếm các kí tự của ảnh có thể là biển số 
                image,
               licensePlateImagesList,
               filteredLicensePlateImagesList,
               licenseBoxList);

            watch.Stop(); //stop the timer
            processTimeLabel.Text = String.Format("Thời gian nhận diện: {0} milli giây", watch.Elapsed.TotalMilliseconds);
            panel1.Controls.Clear();
            Point startPoint = new Point(10, 10);
            //for (int i = 0; i < words.Count; i++)
            //{
            //    List<IInputArray> images = new List<IInputArray>();
            //    //CvInvoke.Imshow("plateROI", _licensePlateDetector.plateROI);
            //    IInputOutputArray plateDrawed = _licensePlateDetector.Segmentation(licensePlateImagesList[i]);
            //    images.Add(_licensePlateDetector.plateROI);
            //    images.Add(licensePlateImagesList[i]);
            //    images.Add(filteredLicensePlateImagesList[i]);
            //    images.Add(plateDrawed);

            //    //CvInvoke.Imshow("licensePlateImages", licensePlateImagesList[i]);
            //    //CvInvoke.Imshow("filteredLicensePlateImages", filteredLicensePlateImagesList[i]);
            //    //CvInvoke.Imshow("plateDrawed", plateDrawed);
            //    AddTextBoxAndImage(
            //       ref startPoint,
            //       String.Format("License: {0}", words[i]),
            //       images);
            //    PointF[] verticesF = licenseBoxList[i].GetVertices();
            //    Point[] vertices = Array.ConvertAll(verticesF, Point.Round);
            //    using (VectorOfPoint pts = new VectorOfPoint(vertices))
            //        CvInvoke.Polylines(image, pts, true, new Bgr(Color.Red).MCvScalar, 2);
            //    //break;
            //}
            List<IInputArray> images = new List<IInputArray>();
            if (_licensePlateDetector.listWordsLongest.Count == 0)
            {
                MessageBox.Show("Lỗi không thể nhận dạng!\nMời bạn chọn ảnh khác!", "Không thể nhận dạng được biển số!", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return; 
            }
            int index = words.IndexOf(_licensePlateDetector.listWordsLongest[_licensePlateDetector.listWordsLongest.Count - 1]);//vị trí ảnh có số lượng kí tự dài nhất (có thể là ảnh biển số)
            IInputOutputArray plateDrawed = _licensePlateDetector.Segmentation(licensePlateImagesList[index]);// vẽ vị trí 
            images.Add(_licensePlateDetector.listPlateROIs[_licensePlateDetector.listWordsLongest[_licensePlateDetector.listWordsLongest.Count-1]]);
            images.Add(licensePlateImagesList[index]);
            images.Add(filteredLicensePlateImagesList[index]);
            images.Add(plateDrawed);
            AddTextBoxAndImage(
                   ref startPoint,
                   String.Format("Biển số: {0}", _licensePlateDetector.listWordsLongest[_licensePlateDetector.listWordsLongest.Count - 1]),
                   images);

            // Khoanh vùng đỏ biển số xe
            PointF[] verticesF = licenseBoxList[index].GetVertices();
            Point[] vertices = Array.ConvertAll(verticesF, Point.Round);
            using (VectorOfPoint pts = new VectorOfPoint(vertices))
                CvInvoke.Polylines(image, pts, true, new Bgr(Color.Red).MCvScalar, 2);

            _licensePlateDetector.listWordsLongest.Clear();
            _licensePlateDetector.listPlateROIs.Clear();
        }

        private void AddTextBoxAndImage(ref Point startPoint, String textBoxText, List<IInputArray> images)
        {
            //Label label = new Label();
            //panel1.Controls.Add(label);
            //label.Text = labelText;
            //label.Width = 100;
            //label.Height = 30;
            //label.Location = startPoint;
            TextBox textBox = new TextBox();
            panel1.Controls.Add(textBox);
            textBox.Text = textBoxText;
            textBox.Width = 200;
            textBox.Height = 200;
            textBox.Location = startPoint;
            textBox.ReadOnly = true;
            startPoint.Y += 2 * textBox.Height;

            for (int i = 0; i < images.Count; i++)
            {
                ImageBox box = new ImageBox();
                panel1.Controls.Add(box);
                using (InputArray iaImage = images[i].GetInputArray())
                {
                    box.ClientSize = iaImage.GetSize();
                    box.Image = images[i];
                    box.Location = startPoint;
                    startPoint.Y += box.Height + 10;
                }
            }
        }

        private void button1_Click(object sender, EventArgs e)
        {
            DialogResult result = openFileDialog1.ShowDialog();
            if (result == DialogResult.OK)
            {
                Mat img;
                try
                {
                    img = CvInvoke.Imread(openFileDialog1.FileName);
                    txtFileDirectory.Text = openFileDialog1.FileName;
                }
                catch
                {
                    MessageBox.Show(String.Format("Tệp không hợp lệ: {0}", openFileDialog1.FileName));
                    return;
                }

                UMat uImg = img.GetUMat(AccessType.ReadWrite);
                imageBox1.Image = uImg;
                ProcessImage(uImg);
            }
        }
    }

}