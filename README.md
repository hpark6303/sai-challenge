# ScienceON AI (SAI) Challenge - RAG Pipeline Project

## 🎯 Project Overview

This project implements a sophisticated Retrieval-Augmented Generation (RAG) pipeline for the ScienceON AI Challenge, designed to answer complex questions using Korean academic papers from the ScienceON database.

## 🚀 Key Features

### **Advanced RAG Pipeline**
- **Multi-stage Processing**: Document retrieval → Semantic filtering → Re-ranking → Answer generation
- **Batch Processing**: Optimized for processing 50 questions efficiently (9.1 minutes total)
- **Bilingual Support**: Korean and English question processing
- **Fallback Mechanisms**: Robust error handling with fallback answer generation

### **Kaggle Competition Ready**
- **Perfect Format Compliance**: Outputs submission.csv matching exact Kaggle requirements
- **50 Article Retrieval**: Extracts top 50 relevant articles per question
- **Null Value Prevention**: Ensures no null values that cause submission errors
- **High Success Rate**: 96% success rate (48/50 questions processed successfully)

### **Performance Optimizations**
- **Memory Efficient**: In-memory storage for fast processing
- **API Rate Limiting**: Intelligent throttling to prevent quota issues
- **Parallel Processing**: Batch operations for improved efficiency
- **Error Recovery**: Graceful handling of API failures

## 📁 Project Structure

```
sai-challenge/
├── rdgenai-api-sample/
│   ├── submission_pipeline_v9_kaggle_format.py  # Main pipeline (refactored)
│   ├── submission_pipeline_v8_batch_fixed.py    # Previous working version
│   ├── submission_pipeline_v8_simple.py         # Simplified version
│   ├── test.csv                                 # Test dataset
│   ├── submission.csv                           # Final Kaggle submission
│   ├── configs/                                 # API credentials
│   │   ├── scienceon_api_credentials.json
│   │   └── gemini_api_credentials.json
│   ├── scienceon_api_example.py                # ScienceON API client
│   ├── gemini_client.py                        # Gemini API client
│   └── requirements.txt                        # Python dependencies
├── README.md                                   # This file
└── .gitignore                                 # Git ignore rules
```

## 🔧 Technical Architecture

### **Core Components**

1. **Document Retrieval**
   - ScienceON API integration for Korean academic papers
   - Dynamic keyword extraction (Korean/English)
   - Synonym expansion for better coverage

2. **Semantic Filtering**
   - Keyword-based relevance scoring
   - Title and abstract analysis
   - Technical term matching

3. **Answer Generation**
   - Gemini API for high-quality responses
   - Structured prompt engineering
   - Fallback answer generation

4. **Kaggle Format Compliance**
   - test.csv as base DataFrame
   - Prediction column for answers
   - 50 prediction_retrieved_article_name columns

### **Pipeline Versions**

| Version | Key Features | Performance | Status |
|---------|-------------|-------------|---------|
| v7 | Initial sequential processing | ~120 minutes | Baseline |
| v8_simple | Simplified batch processing | ~15 minutes | Intermediate |
| v8_batch_fixed | Advanced filtering | ~7 minutes | Working |
| **v9_kaggle_format** | **Perfect Kaggle compliance** | **~9 minutes** | **Production** |

## 📊 Performance Metrics

### **Latest Results (v9)**
- **Total Processing Time**: 543.68 seconds (9.1 minutes)
- **Average Time per Question**: 10.87 seconds
- **Success Rate**: 96% (48/50 questions)
- **File Size**: 50 rows × 107 columns
- **Null Values**: 0 (perfect for Kaggle submission)

### **Improvement Timeline**
- **v6-v7**: Sequential processing (inefficient)
- **v8**: Batch processing implementation
- **v9**: Kaggle format compliance + optimizations

## 🛠️ Installation & Setup

### **Prerequisites**
```bash
# Python 3.10+ required
conda create -n llm2024 python=3.10
conda activate llm2024
```

### **Dependencies**
```bash
pip install -r rdgenai-api-sample/requirements.txt
```

### **API Configuration**
1. Create `configs/scienceon_api_credentials.json`:
```json
{
  "auth_key": "your_scienceon_auth_key",
  "client_id": "your_client_id", 
  "mac_address": "your_mac_address"
}
```

2. Create `configs/gemini_api_credentials.json`:
```json
{
  "api_key": "your_gemini_api_key"
}
```

## 🚀 Usage

### **Run the Pipeline**
```bash
cd rdgenai-api-sample
python submission_pipeline_v9_kaggle_format.py
```

### **Kaggle Submission**
```bash
kaggle competitions submit -c sai-challenge -f submission.csv -m "Your submission message"
```

## 🔍 Key Technical Decisions

### **1. Batch Processing Implementation**
- **Problem**: Sequential processing was too slow (~120 minutes)
- **Solution**: Implemented batch processing for all pipeline stages
- **Result**: 13x speed improvement (9.1 minutes)

### **2. Library Compatibility Issues**
- **Problem**: PyTorch MPS crashes on Apple Silicon
- **Solution**: Forced CPU usage and removed problematic transformers
- **Result**: Stable execution on macOS

### **3. Kaggle Format Compliance**
- **Problem**: Output format didn't match Kaggle requirements
- **Solution**: Refactored to use test.csv as base + new prediction columns
- **Result**: Perfect submission format with 107 columns

### **4. API Quota Management**
- **Problem**: Gemini API quota exceeded during processing
- **Solution**: Implemented fallback answer generation
- **Result**: 96% success rate even with API limitations

## 📈 Development Journey

### **Phase 1: Initial Development**
- Basic RAG pipeline with ScienceON API
- Sequential processing of questions
- Simple answer generation

### **Phase 2: Performance Optimization**
- Implemented batch processing
- Added semantic filtering and re-ranking
- Optimized for speed and efficiency

### **Phase 3: Kaggle Compliance**
- Refactored output format to match requirements
- Added 50 article retrieval per question
- Ensured null value prevention

### **Phase 4: Production Ready**
- Comprehensive error handling
- Robust fallback mechanisms
- Perfect Kaggle submission format

## 🎯 Results & Achievements

### **Competition Performance**
- **Successfully submitted** to ScienceON AI Challenge
- **Perfect format compliance** - no submission errors
- **High-quality answers** with academic paper references
- **Efficient processing** - 9.1 minutes for 50 questions

### **Technical Achievements**
- **13x performance improvement** over baseline
- **96% success rate** with robust error handling
- **Bilingual support** for Korean and English questions
- **Production-ready pipeline** with comprehensive documentation

## 🔮 Future Improvements

### **Potential Enhancements**
1. **Advanced Embeddings**: Implement sentence-transformers for better semantic search
2. **Multi-modal Support**: Add support for images and diagrams
3. **Real-time Processing**: Stream processing for live question answering
4. **Enhanced Prompting**: More sophisticated prompt engineering
5. **Model Fine-tuning**: Custom model training for domain-specific tasks

### **Scalability Considerations**
- **Distributed Processing**: Multi-node processing for larger datasets
- **Caching Layer**: Redis/Memcached for document caching
- **API Optimization**: Connection pooling and request batching
- **Monitoring**: Real-time performance monitoring and alerting

## 📝 License

This project is developed for the ScienceON AI Challenge. Please refer to the competition guidelines for usage terms.

## 🤝 Contributing

This project was developed as part of the ScienceON AI Challenge. For questions or collaboration opportunities, please refer to the competition guidelines.

---

**Developed with ❤️ for the ScienceON AI Challenge**
