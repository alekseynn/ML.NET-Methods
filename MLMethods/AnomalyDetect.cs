using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.TimeSeries;

namespace MLMethods
{
    class AnomalyDetect
    {
        class Program
        {
            static void Main(string[] args)
            {

                MLContext ml = new MLContext();
                var data = LoadDataFromFile(@".\Data\Dataset1.csv");
                //Преобразовать данные в экземпляр IDataView.
                var dataView = ml.Data.LoadFromEnumerable(data);
                //Подготовить переменные для вывода данные
                string outputColumnName = nameof(SrCnnAnomalyDetection.Prediction);
                string inputColumnName = nameof(TimeSeriesData.Value);
                //Perform the batch anomaly detection for each input data point
                //Выполнить обнаружение аномалий для каждой точки входных данных
                // Do batch anomaly detection
                var outputDataView = ml.AnomalyDetection.DetectEntireAnomalyBySrCnn
                    (dataView,
                    outputColumnName,
                    inputColumnName,
                    threshold: 0.30,
                    batchSize: -1,
                    sensitivity: 91,
                    detectMode: SrCnnDetectMode.AnomalyOnly);
                //Получить только что созданный столбец
                var predictionColumn = ml.Data.CreateEnumerable<SrCnnAnomalyDetection>(
                                   outputDataView, reuseRowObject: false);
                //Выполнить итерацию по полученному столбцу, чтобы определить аномалии в данных
                int k = 0;
                foreach (var prediction in predictionColumn)
                {

                    Console.WriteLine($"{data[k].Day}: {data[k].Value:0.0}. Данные DetectEntireAnomalyBySrCnn: isAnomaly: {prediction.Prediction[0]}, RawScore: {prediction.Prediction[1]}, Mag:{prediction.Prediction[2]} ");

                    if (prediction.Prediction[0] > 0)
                    {
                        Console.WriteLine($"Обнаружена аномалия! {data[k].Day}: {data[k].Value:0.0}. Данные DetectEntireAnomalyBySrCnn: isAnomaly: {prediction.Prediction[0]}, RawScore: {prediction.Prediction[1]}, Mag:{prediction.Prediction[2]} ");
                    }
                    k++;
                }
            }
        }
        private class TimeSeriesData
        {
            public double Value { get; set; }
            public string Day { get; set; }
        }

        private class SrCnnAnomalyDetection
        {
            [VectorType]
            public double[] Prediction { get; set; }
        }

        private static List<TimeSeriesData> LoadDataFromFile(string fileName)
        {
            return File.ReadAllLines(fileName)
            .Skip(1)
            .Select(f => new TimeSeriesData()
            {
                Day = f.Split(new char[] { ';' },
            StringSplitOptions.RemoveEmptyEntries)[0],
                Value = Convert.ToDouble(f.Split(new char[] { ';' },
            StringSplitOptions.RemoveEmptyEntries)[1])
            })
            .ToList();
        }

        private static List<TimeSeriesData> LoadDataFromFile2(string fileName, int column)
        {
            return File.ReadAllLines(fileName)
            .Skip(1)
            .Select(f => new TimeSeriesData()
            {
                Day = f.Split(new char[] { ';' },
            StringSplitOptions.RemoveEmptyEntries)[column]
            })
            .ToList();
        }
    }


}
