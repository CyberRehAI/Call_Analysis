import { BarChart3, Mic, Brain, TrendingUp, Shield, Zap } from 'lucide-react';

export function AboutSection() {
  const features = [
    {
      icon: Mic,
      title: 'Audio Transcription',
      description:
        'Convert speech to text using advanced Whisper models for accurate transcription.',
    },
    {
      icon: Brain,
      title: 'Sentiment Analysis',
      description:
        'Analyze emotional tone and sentiment throughout the conversation using BERT models.',
    },
    {
      icon: TrendingUp,
      title: 'Sales Prediction',
      description:
        'Predict the probability of a successful sale based on conversation patterns.',
    },
    {
      icon: BarChart3,
      title: 'Interactive Dashboard',
      description: 'Visualize insights through charts, graphs, and real-time analytics.',
    },
    {
      icon: Shield,
      title: 'PII Protection',
      description: 'Automatically detect and mask personally identifiable information.',
    },
    {
      icon: Zap,
      title: 'Real-time Processing',
      description: 'Get instant results with our optimized processing pipeline.',
    },
  ];

  const stats = [
    { label: 'Supported Formats', value: 'WAV, MP3, M4A' },
    { label: 'Max File Size', value: '100MB' },
    { label: 'Processing Time on GPU', value: '< 2 minutes' },
    { label: 'Accuracy', value: '80%+' },
  ];

  return (
    <div className="space-y-12">
      {/* Hero Section */}
      <div className="text-center">
        <h2 className="text-4xl font-bold text-gray-100 mb-4">
          About Call Analysis System
        </h2>
        <p className="text-xl text-gray-200 max-w-3xl mx-auto">
          A comprehensive multimodal sentiment analysis platform that transforms telephonic
          conversations into actionable business insights through advanced AI and machine
          learning.
        </p>
      </div>

      {/* Features Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {features.map((feature, index) => (
          <div key={index} className="card text-center">
            <div className="flex justify-center mb-4">
              <div className="p-3 bg-primary-100 rounded-full">
                <feature.icon className="h-8 w-8 text-primary-600" />
              </div>
            </div>
            <h3 className="text-lg font-semibold text-gray-100 mb-2">
              {feature.title}
            </h3>
            <p className="text-gray-200">{feature.description}</p>
          </div>
        ))}
      </div>

      {/* Stats Section */}
      <div className="card">
        <h3 className="text-2xl font-bold text-gray-100 mb-6 text-center">
          System Specifications
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
          {stats.map((stat, index) => (
            <div key={index} className="text-center">
              <div className="text-3xl font-bold text-primary-600 mb-2">
                {stat.value}
              </div>
              <div className="text-sm text-gray-200">{stat.label}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Technology Stack */}
      <div className="card">
        <h3 className="text-2xl font-bold text-gray-100 mb-6">
          Technology Stack
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div>
            <h4 className="text-lg font-semibold text-gray-100 mb-4">Backend</h4>
            <ul className="space-y-2 text-gray-200">
              <li>• Python Flask - Web framework</li>
              <li>• MongoDB Atlas - Database</li>
              <li>• Hugging Face Transformers - AI models</li>
              <li>• Whisper - Speech recognition</li>
              <li>• BERT/DistilBERT - Sentiment analysis</li>
              <li>• PyAnnote - Speaker diarization</li>
            </ul>
          </div>
          <div>
            <h4 className="text-lg font-semibold text-gray-100 mb-4">Frontend</h4>
            <ul className="space-y-2 text-gray-200">
              <li>• Next.js 14 - React framework</li>
              <li>• TypeScript - Type safety</li>
              <li>• TailwindCSS - Styling</li>
              <li>• Recharts - Data visualization</li>
              <li>• React Hot Toast - Notifications</li>
              <li>• Lucide React - Icons</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Use Cases */}
      <div className="card">
        <h3 className="text-2xl font-bold text-gray-100 mb-6">Use Cases</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="text-lg font-semibold text-gray-100 mb-3">
              Sales & Marketing
            </h4>
            <ul className="space-y-2 text-gray-200">
              <li>• Analyze sales call effectiveness</li>
              <li>• Identify successful conversation patterns</li>
              <li>• Train sales teams with data-driven insights</li>
              <li>• Predict deal closure probability</li>
            </ul>
          </div>
          <div>
            <h4 className="text-lg font-semibold text-gray-100 mb-3">
              Customer Service
            </h4>
            <ul className="space-y-2 text-gray-200">
              <li>• Monitor customer satisfaction</li>
              <li>• Detect frustrated customers early</li>
              <li>• Improve response strategies</li>
              <li>• Quality assurance and training</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Call to Action */}
      <div className="card text-center">
        <h3 className="text-2xl font-bold text-gray-100 mb-4">Get Started</h3>
        <p className="text-gray-200 mb-6">
          Ready to transform your call analysis? Upload your first audio file and start
          gaining insights today.
        </p>
        <a href="/#home" className="btn-primary">
          Start Analyzing
        </a>
      </div>
    </div>
  );
}


