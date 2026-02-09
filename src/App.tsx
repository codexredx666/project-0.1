import { useState, useCallback } from 'react'
import { Upload, Image as ImageIcon, Brain, Cpu, Zap, Database } from 'lucide-react'
import { Button } from './components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './components/ui/card'
import { Badge } from './components/ui/badge'
import { Progress } from './components/ui/progress'
import { toast } from 'sonner'

interface PredictionResult {
  prediction: string
  confidence: number
  probabilities: {
    Bike: number
    Car: number
  }
}

function App() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null)
  const [predicting, setPredicting] = useState(false)
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [isLocalAPI, setIsLocalAPI] = useState(false)
  const [apiChecked, setApiChecked] = useState(false)

  const handleImageUpload = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    if (!file.type.startsWith('image/')) {
      toast.error('Please upload an image file')
      return
    }

    const reader = new FileReader()
    reader.onload = (event) => {
      setSelectedImage(event.target?.result as string)
      setResult(null)
    }
    reader.readAsDataURL(file)
  }, [])

  const handlePredict = useCallback(async () => {
    if (!selectedImage) {
      toast.error('Please upload an image first')
      return
    }

    setPredicting(true)
    setResult(null)

    try {
      // Check if running locally or in production
      const isLocalhost = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
      
      if (isLocalhost) {
        // Try to connect to local Python API
        try {
          const response = await fetch('http://localhost:5000/api/predict', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              image: selectedImage,
            }),
          })

          if (!response.ok) {
            throw new Error('Prediction failed')
          }

          const data: PredictionResult = await response.json()
          setResult(data)
          setIsLocalAPI(true)
          setApiChecked(true)
          toast.success(`Prediction: ${data.prediction} (${(data.confidence * 100).toFixed(1)}% confident)`)
          return
        } catch (error) {
          console.warn('Local API not available, using demo mode')
          setIsLocalAPI(false)
          setApiChecked(true)
          toast.info('Local API not running. Using demo mode.')
        }
      } else {
        setIsLocalAPI(false)
        setApiChecked(true)
      }
      
      // Demo mode for production or when local API is not available
      // Simulate AI inference with basic image analysis
      const result = await simulatePrediction(selectedImage)
      setResult(result)
      toast.success(`Demo Prediction: ${result.prediction} (${(result.confidence * 100).toFixed(1)}% confident)`)
      
    } catch (error) {
      console.error('Prediction error:', error)
      toast.error('Prediction failed. Please try again.')
    } finally {
      setPredicting(false)
    }
  }, [selectedImage])

  // Simulate prediction for demo purposes
  const simulatePrediction = async (imageData: string): Promise<PredictionResult> => {
    // Add realistic delay
    await new Promise(resolve => setTimeout(resolve, 1500))
    
    // Analyze image characteristics
    const img = new Image()
    img.src = imageData
    
    return new Promise((resolve) => {
      img.onload = () => {
        // Simple heuristic based on image characteristics
        const canvas = document.createElement('canvas')
        canvas.width = 100
        canvas.height = 100
        const ctx = canvas.getContext('2d')
        
        if (ctx) {
          ctx.drawImage(img, 0, 0, 100, 100)
          const imageData = ctx.getImageData(0, 0, 100, 100)
          const data = imageData.data
          
          // Calculate average brightness and color variance
          let r = 0, g = 0, b = 0
          for (let i = 0; i < data.length; i += 4) {
            r += data[i]
            g += data[i + 1]
            b += data[i + 2]
          }
          const pixelCount = data.length / 4
          r /= pixelCount
          g /= pixelCount
          b /= pixelCount
          
          // Simple heuristic: darker images tend to be cars, lighter tend to be bikes
          // This is just for demo purposes
          const brightness = (r + g + b) / 3
          const isCar = brightness < 130
          
          const carProb = isCar ? 0.75 + Math.random() * 0.2 : 0.15 + Math.random() * 0.2
          const bikeProb = 1 - carProb
          
          resolve({
            prediction: isCar ? 'Car' : 'Bike',
            confidence: Math.max(carProb, bikeProb),
            probabilities: {
              Bike: bikeProb,
              Car: carProb
            }
          })
        } else {
          // Fallback random prediction
          const isCar = Math.random() > 0.5
          const conf = 0.65 + Math.random() * 0.25
          resolve({
            prediction: isCar ? 'Car' : 'Bike',
            confidence: conf,
            probabilities: {
              Bike: isCar ? 1 - conf : conf,
              Car: isCar ? conf : 1 - conf
            }
          })
        }
      }
      
      img.onerror = () => {
        // Fallback random prediction
        const isCar = Math.random() > 0.5
        const conf = 0.65 + Math.random() * 0.25
        resolve({
          prediction: isCar ? 'Car' : 'Bike',
          confidence: conf,
          probabilities: {
            Bike: isCar ? 1 - conf : conf,
            Car: isCar ? conf : 1 - conf
          }
        })
      }
    })
  }

  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader()
      reader.onload = (event) => {
        setSelectedImage(event.target?.result as string)
        setResult(null)
      }
      reader.readAsDataURL(file)
    }
  }, [])

  const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
  }, [])

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted">
      {/* Header */}
      <header className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary">
                <Brain className="h-6 w-6 text-primary-foreground" />
              </div>
              <div>
                <h1 className="text-2xl font-bold">Bike vs Car Classifier</h1>
                <p className="text-sm text-muted-foreground">CNN Model with GPU Training</p>
              </div>
            </div>
            <div className="flex gap-2">
              <Badge variant="secondary" className="gap-1">
                <Cpu className="h-3 w-3" />
                GPU Ready
              </Badge>
              <Badge variant="secondary" className="gap-1">
                <Zap className="h-3 w-3" />
                TensorFlow
              </Badge>
              {apiChecked && (
                <Badge variant={isLocalAPI ? "default" : "outline"} className="gap-1">
                  <div className={`h-2 w-2 rounded-full ${isLocalAPI ? 'bg-green-500' : 'bg-yellow-500'}`} />
                  {isLocalAPI ? 'Local API' : 'Demo Mode'}
                </Badge>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        {/* Feature Cards */}
        <div className="mb-8 grid gap-4 md:grid-cols-3">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 text-base">
                <Brain className="h-4 w-4" />
                Deep Learning
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">
                Custom CNN with 4 convolutional blocks and batch normalization
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 text-base">
                <Cpu className="h-4 w-4" />
                GPU Accelerated
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">
                TensorFlow/Keras with CUDA support for fast training
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 text-base">
                <Database className="h-4 w-4" />
                Production Ready
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">
                Complete pipeline from training to deployment
              </p>
            </CardContent>
          </Card>
        </div>

        {/* Main Prediction Area */}
        <div className="grid gap-6 lg:grid-cols-2">
          {/* Upload Section */}
          <Card>
            <CardHeader>
              <CardTitle>Upload Image</CardTitle>
              <CardDescription>
                Upload a bike or car image for classification
                {!apiChecked && ' • Checking API status...'}
                {apiChecked && !isLocalAPI && ' • Demo mode active (simplified predictions)'}
                {apiChecked && isLocalAPI && ' • Connected to local trained model'}
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Drop Zone */}
              <div
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                className="flex min-h-[300px] cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-muted-foreground/25 bg-muted/50 p-8 transition-colors hover:border-muted-foreground/50 hover:bg-muted"
                onClick={() => document.getElementById('file-input')?.click()}
              >
                {selectedImage ? (
                  <div className="relative w-full">
                    <img
                      src={selectedImage}
                      alt="Selected"
                      className="mx-auto max-h-[280px] rounded-lg object-contain"
                    />
                  </div>
                ) : (
                  <div className="text-center">
                    <Upload className="mx-auto h-12 w-12 text-muted-foreground" />
                    <p className="mt-4 text-sm font-medium">Drop an image here</p>
                    <p className="mt-1 text-xs text-muted-foreground">or click to browse</p>
                    <p className="mt-2 text-xs text-muted-foreground">Supports: JPG, PNG, JPEG</p>
                  </div>
                )}
              </div>

              <input
                id="file-input"
                type="file"
                accept="image/*"
                onChange={handleImageUpload}
                className="hidden"
              />

              {/* Action Buttons */}
              <div className="flex gap-2">
                <Button
                  onClick={handlePredict}
                  disabled={!selectedImage || predicting}
                  className="flex-1"
                  size="lg"
                >
                  {predicting ? (
                    <>Processing...</>
                  ) : (
                    <>
                      <Brain className="mr-2 h-4 w-4" />
                      Predict
                    </>
                  )}
                </Button>
                {selectedImage && (
                  <Button
                    onClick={() => {
                      setSelectedImage(null)
                      setResult(null)
                    }}
                    variant="outline"
                    size="lg"
                  >
                    Clear
                  </Button>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Results Section */}
          <Card>
            <CardHeader>
              <CardTitle>Prediction Results</CardTitle>
              <CardDescription>Model classification output</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {result ? (
                <>
                  {/* Main Result */}
                  <div className="rounded-lg bg-muted/50 p-6 text-center">
                    <div className="mb-2 flex items-center justify-center gap-2">
                      <ImageIcon className="h-8 w-8 text-primary" />
                    </div>
                    <h3 className="mb-1 text-3xl font-bold">{result.prediction}</h3>
                    <p className="text-sm text-muted-foreground">
                      Confidence: {(result.confidence * 100).toFixed(2)}%
                    </p>
                  </div>

                  {/* Detailed Probabilities */}
                  <div className="space-y-4">
                    <h4 className="font-semibold">Class Probabilities</h4>

                    {/* Bike */}
                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-sm">
                        <span className="font-medium">🚴 Bike</span>
                        <span className="text-muted-foreground">
                          {(result.probabilities.Bike * 100).toFixed(2)}%
                        </span>
                      </div>
                      <Progress value={result.probabilities.Bike * 100} />
                    </div>

                    {/* Car */}
                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-sm">
                        <span className="font-medium">🚗 Car</span>
                        <span className="text-muted-foreground">
                          {(result.probabilities.Car * 100).toFixed(2)}%
                        </span>
                      </div>
                      <Progress value={result.probabilities.Car * 100} />
                    </div>
                  </div>

                  {/* Model Info */}
                  <div className="rounded-lg border bg-card p-4 text-sm">
                    <p className="mb-1 font-medium">Model Information</p>
                    <p className="text-muted-foreground">Architecture: Custom CNN</p>
                    <p className="text-muted-foreground">Input Size: 224x224x3</p>
                    <p className="text-muted-foreground">Framework: TensorFlow/Keras</p>
                  </div>
                </>
              ) : (
                <div className="flex min-h-[400px] flex-col items-center justify-center text-center text-muted-foreground">
                  <ImageIcon className="mb-4 h-16 w-16" />
                  <p className="text-sm">Upload an image and click Predict</p>
                  <p className="mt-1 text-xs">to see classification results</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Instructions */}
        <Card className="mt-8">
          <CardHeader>
            <CardTitle>Getting Started</CardTitle>
            <CardDescription>Complete setup and training instructions</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <h4 className="mb-2 font-semibold">1. Install Python Dependencies</h4>
              <code className="block rounded bg-muted p-2 text-sm">
                cd python && pip install -r requirements.txt
              </code>
            </div>

            <div>
              <h4 className="mb-2 font-semibold">2. Prepare Dataset</h4>
              <code className="block rounded bg-muted p-2 text-sm">
                python dataset.py
              </code>
              <p className="mt-1 text-xs text-muted-foreground">
                Place bike and car images in data/train/, data/validation/, and data/test/ folders
              </p>
            </div>

            <div>
              <h4 className="mb-2 font-semibold">3. Train Model (with GPU)</h4>
              <code className="block rounded bg-muted p-2 text-sm">
                python train.py --epochs 50 --batch_size 32
              </code>
              <p className="mt-1 text-xs text-muted-foreground">
                Model will be saved to models/best_model.keras
              </p>
            </div>

            <div>
              <h4 className="mb-2 font-semibold">4. Start API Server</h4>
              <code className="block rounded bg-muted p-2 text-sm">
                python api.py
              </code>
              <p className="mt-1 text-xs text-muted-foreground">
                API runs on http://localhost:5000
              </p>
            </div>

            <div>
              <h4 className="mb-2 font-semibold">5. Use This Web Interface</h4>
              <p className="text-sm text-muted-foreground">
                Upload images and get real-time predictions from your trained model
              </p>
            </div>
          </CardContent>
        </Card>
      </main>

      {/* Footer */}
      <footer className="mt-12 border-t bg-muted/50 py-6">
        <div className="container mx-auto px-4 text-center text-sm text-muted-foreground">
          <p>
            CNN Model Project · TensorFlow/Keras · GPU Accelerated · Production Ready
          </p>
        </div>
      </footer>
    </div>
  )
}

export default App 