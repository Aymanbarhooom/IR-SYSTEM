# services/query_refiner.py

import re
import nltk
import ssl
from typing import List, Tuple, Optional, Set
from collections import Counter
import json

# تحميل موارد NLTK تلقائياً
def download_nltk_resources():
    """تحميل جميع موارد NLTK المطلوبة"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        nltk.download('punkt_tab')
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
        nltk.download('omw-1.4')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

download_nltk_resources()

from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob


class QueryRefiner:
    """
    تحسين الاستعلامات: تصحيح إملائي + توسيع مرادفات ذكي
    """
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.common_words = self._load_common_words()
        self.synonym_cache = {}
        self.word_freq_cache = {}
        
    def _load_common_words(self) -> Set[str]:
        """تحميل الكلمات الشائعة لتجنب توسيعها"""
        return {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
            'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their'
        }
    
    # ========== SPELLING CORRECTION ==========
    
    def correct_spelling(self, query: str) -> Tuple[str, List[Tuple[str, str]]]:
        """
        تصحيح الأخطاء الإملائية باستخدام TextBlob
        
        Returns:
            (query_corrected, corrections_list)
        """
        blob = TextBlob(query)
        corrected_words = []
        corrections = []
        
        for word in blob.words:
            if word.string.lower() not in self.stop_words:
                corrected = word.correct()
                if corrected.string != word.string and corrected.string.isalpha():
                    corrections.append((word.string, corrected.string))
                    corrected_words.append(corrected.string)
                else:
                    corrected_words.append(word.string)
            else:
                corrected_words.append(word.string)
        
        corrected_query = ' '.join(corrected_words)
        
        if corrections:
            print(f"✏️ Spelling corrections: {corrections}")
        
        return corrected_query, corrections
    
    # ========== SMART SYNONYM EXPANSION ==========
    
    def _get_word_synonyms(self, word: str, max_synonyms: int = 2) -> List[str]:
        """
        الحصول على مرادفات ذكية لكلمة معينة
        """
        # التحقق من الكاش
        cache_key = f"{word}_{max_synonyms}"
        if cache_key in self.synonym_cache:
            return self.synonym_cache[cache_key]
        
        # تجاهل الكلمات الشائعة والقصيرة
        if len(word) < 3 or word in self.common_words or word in self.stop_words:
            return []
        
        synonyms = set()
        
        # الحصول على المرادفات من WordNet
        for synset in wn.synsets(word):
            # تصفية حسب جزء الكلام (noun, verb, adj)
            for lemma in synset.lemmas():
                synonym = lemma.name().replace('_', ' ')
                
                # شروط القبول:
                # 1. ليس نفس الكلمة الأصلية
                # 2. يحتوي على حروف فقط
                # 3. ليس كلمة شائعة
                # 4. طول مناسب
                if (synonym.lower() != word.lower() and 
                    synonym.isalpha() and 
                    len(synonym) >= 3 and
                    synonym.lower() not in self.common_words and
                    synonym.lower() not in self.stop_words):
                    
                    # إضافة المرادف مع التحقق من أنه ليس مرادفاً لكلمة شائعة
                    synonyms.add(synonym.lower())
        
        # تصفية المرادفات التي قد تكون ضوضاء
        filtered_synonyms = []
        for syn in synonyms:
            # تجاهل المرادفات التي تبدأ بنفس الحرف وتشبه الكلمة الأصلية كثيراً
            if syn.startswith(word[:2]) and len(syn) <= len(word) + 2:
                continue
            filtered_synonyms.append(syn)
        
        # اختيار أفضل المرادفات
        final_synonyms = self._rank_synonyms(word, filtered_synonyms)[:max_synonyms]
        
        # حفظ في الكاش
        self.synonym_cache[cache_key] = final_synonyms
        
        return final_synonyms
    
    def _rank_synonyms(self, original_word: str, synonyms: List[str]) -> List[str]:
        """
        ترتيب المرادفات حسب الجودة والأهمية
        """
        if not synonyms:
            return []
        
        # حساب درجة لكل مرادف
        ranked = []
        for syn in synonyms:
            score = 0
            
            # 1. تفضيل المرادفات الأكثر شيوعاً (تخمين بناءً على الطول)
            if len(syn) >= len(original_word):
                score += 1
            
            # 2. تفضيل المرادفات التي تبدأ بحرف مختلف (أكثر تنوعاً)
            if syn[0] != original_word[0]:
                score += 1
            
            # 3. تفضيل المرادفات التي لا تحتوي على أرقام أو رموز
            if syn.isalpha():
                score += 1
            
            ranked.append((syn, score))
        
        # ترتيب حسب الدرجة
        ranked.sort(key=lambda x: x[1], reverse=True)
        
        return [syn for syn, _ in ranked]
    
    def expand_synonyms(self, query: str, max_synonyms_per_word: int = 2, 
                        min_word_length: int = 4) -> Tuple[str, dict]:
        """
        توسيع الاستعلام بإضافة مرادفات ذكية
        
        Returns:
            (expanded_query, expansion_info)
        """
        # تقسيم الاستعلام إلى كلمات
        words = word_tokenize(query.lower())
        
        # تصفية الكلمات
        important_words = [
            w for w in words 
            if w.isalpha() and 
            len(w) >= min_word_length and 
            w not in self.stop_words and
            w not in self.common_words
        ]
        
        if not important_words:
            return query, {}
        
        # تجميع الكلمات الأصلية والمرادفات
        expanded_parts = []
        expansion_info = {}
        
        for word in important_words:
            # الحصول على مرادفات ذكية
            synonyms = self._get_word_synonyms(word, max_synonyms_per_word)
            
            if synonyms:
                expansion_info[word] = synonyms
                # إضافة الكلمة الأصلية + مرادفاتها
                expanded_parts.append(word)
                expanded_parts.extend(synonyms)
            else:
                expanded_parts.append(word)
        
        # إضافة الكلمات غير المهمة كما هي
        for word in words:
            if word not in important_words and word not in expansion_info:
                expanded_parts.append(word)
        
        # إزالة التكرارات مع الحفاظ على الترتيب
        seen = set()
        unique_parts = []
        for part in expanded_parts:
            if part not in seen:
                seen.add(part)
                unique_parts.append(part)
        
        expanded_query = ' '.join(unique_parts)
        
        if expansion_info:
            print(f"🔍 Synonym expansion: {expansion_info}")
        
        return expanded_query, expansion_info
    
    # ========== QUERY WEIGHTING (OPTIONAL) ==========
    
    def query_weighting(self, query: str, important_terms: List[str] = None, 
                        boost_factor: float = 2.0) -> str:
        """
        تثقيل الكلمات المهمة في الاستعلام
        
        Args:
            query: الاستعلام الأصلي
            important_terms: قائمة الكلمات المهمة (إذا كانت None، سيتم استخراجها)
            boost_factor: عامل التثقيل
        """
        words = query.split()
        
        if important_terms is None:
            # استخراج الكلمات المهمة تلقائياً (الكلمات التي ليست شائعة)
            important_terms = [
                w for w in words 
                if w.isalpha() and 
                len(w) >= 4 and 
                w not in self.stop_words and
                w not in self.common_words
            ]
        
        weighted_parts = []
        for word in words:
            if word in important_terms:
                weighted_parts.append(f"{word}^{boost_factor:.1f}")
            else:
                weighted_parts.append(word)
        
        return ' '.join(weighted_parts)
    
    # ========== QUERY EXPANSION WITH CONTEXT ==========
    
    def expand_with_context(self, query: str, context_docs: List[str] = None,
                           max_terms: int = 5) -> str:
        """
        توسيع الاستعلام باستخدام سياق الوثائق (Feedback-based expansion)
        
        Args:
            query: الاستعلام الأصلي
            context_docs: قائمة بالوثائق ذات الصلة
            max_terms: عدد المصطلحات الإضافية
        """
        if not context_docs:
            return query
        
        # استخراج المصطلحات المهمة من الوثائق
        all_terms = []
        for doc in context_docs[:5]:  # استخدام أول 5 وثائق فقط
            words = word_tokenize(doc.lower())
            terms = [
                w for w in words 
                if w.isalpha() and 
                len(w) >= 4 and 
                w not in self.stop_words and
                w not in self.common_words
            ]
            all_terms.extend(terms)
        
        if not all_terms:
            return query
        
        # حساب تكرار المصطلحات
        term_freq = Counter(all_terms)
        
        # استخراج مصطلحات الاستعلام الأصلي
        query_terms = set(word_tokenize(query.lower()))
        
        # اختيار المصطلحات الأكثر تكراراً التي ليست في الاستعلام الأصلي
        new_terms = []
        for term, freq in term_freq.most_common(max_terms * 2):
            if term not in query_terms:
                new_terms.append(term)
                if len(new_terms) >= max_terms:
                    break
        
        if new_terms:
            expanded_query = f"{query} {' '.join(new_terms)}"
            print(f"📚 Context expansion added: {new_terms}")
            return expanded_query
        
        return query
    
    # ========== FULL REFINEMENT PIPELINE ==========
    
    def refine(self, query: str, 
               use_spelling: bool = True,
               use_synonyms: bool = True,
               use_weighting: bool = False,
               use_context: bool = False,
               context_docs: List[str] = None,
               max_synonyms: int = 2,
               important_terms: List[str] = None,
               boost_factor: float = 2.0) -> dict:
        """
        تطبيق جميع تحسينات الاستعلام في خطوة واحدة
        
        Returns:
            dict: {
                'original_query': str,
                'refined_query': str,
                'corrections': List[Tuple],
                'synonyms_added': dict,
                'weighted': bool,
                'context_terms': List[str]
            }
        """
        print("\n" + "="*50)
        print("🔧 QUERY REFINEMENT")
        print("="*50)
        
        result = {
            'original_query': query,
            'refined_query': query,
            'corrections': [],
            'synonyms_added': {},
            'weighted': False,
            'context_terms': []
        }
        
        current_query = query
        
        # 1. Spelling Correction
        if use_spelling:
            corrected, corrections = self.correct_spelling(current_query)
            current_query = corrected
            result['corrections'] = corrections
        
        # 2. Synonym Expansion
        if use_synonyms:
            expanded, synonyms = self.expand_synonyms(
                current_query, 
                max_synonyms_per_word=max_synonyms
            )
            current_query = expanded
            result['synonyms_added'] = synonyms
        
        # 3. Query Weighting
        if use_weighting:
            current_query = self.query_weighting(
                current_query, 
                important_terms=important_terms,
                boost_factor=boost_factor
            )
            result['weighted'] = True
        
        # 4. Context Expansion
        if use_context and context_docs:
            context_query = self.expand_with_context(
                current_query, 
                context_docs
            )
            # استخراج المصطلحات المضافة
            original_terms = set(word_tokenize(current_query.lower()))
            new_terms = set(word_tokenize(context_query.lower())) - original_terms
            result['context_terms'] = list(new_terms)
            current_query = context_query
        
        result['refined_query'] = current_query
        
        # عرض التقرير
        print(f"\n📝 Original: {query}")
        print(f"✨ Refined:  {current_query}")
        if result['corrections']:
            print(f"✏️ Corrections: {result['corrections']}")
        if result['synonyms_added']:
            print(f"🔍 Synonyms: {result['synonyms_added']}")
        if result['context_terms']:
            print(f"📚 Context: {result['context_terms']}")
        print("="*50)
        
        return result


class QueryHistory:
    """
    تتبع سجل الاستعلامات لتحسين النتائج المستقبلية
    """
    
    def __init__(self, history_file: str = "query_history.json"):
        self.history_file = history_file
        self.history = []
        self.load_history()
    
    def load_history(self):
        """تحميل سجل الاستعلامات"""
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                self.history = json.load(f)
        except FileNotFoundError:
            self.history = []
    
    def save_history(self):
        """حفظ سجل الاستعلامات"""
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)
    
    def add_query(self, query: str, clicked_docs: List[str] = None, 
                  results: List[str] = None):
        """إضافة استعلام جديد إلى السجل"""
        self.history.append({
            'query': query,
            'timestamp': str(__import__('datetime').datetime.now()),
            'clicked_docs': clicked_docs or [],
            'results_count': len(results) if results else 0
        })
        self.save_history()
    
    def get_popular_terms(self, limit: int = 5) -> List[str]:
        """الحصول على المصطلحات الأكثر شيوعاً في سجل البحث"""
        if not self.history:
            return []
        
        all_terms = []
        for entry in self.history:
            all_terms.extend(word_tokenize(entry['query'].lower()))
        
        # تصفية الكلمات الشائعة
        stop_words = set(stopwords.words('english'))
        filtered_terms = [
            t for t in all_terms 
            if t.isalpha() and 
            len(t) >= 3 and 
            t not in stop_words
        ]
        
        if not filtered_terms:
            return []
        
        # ترتيب حسب التكرار
        term_freq = Counter(filtered_terms)
        return [term for term, _ in term_freq.most_common(limit)]
    
    def get_related_queries(self, query: str, limit: int = 3) -> List[str]:
        """الحصول على استعلامات مشابهة من السجل"""
        query_terms = set(word_tokenize(query.lower()))
        
        similar = []
        for entry in self.history:
            entry_terms = set(word_tokenize(entry['query'].lower()))
            overlap = len(query_terms & entry_terms)
            if overlap > 0 and entry['query'] != query:
                similar.append((entry['query'], overlap))
        
        # ترتيب حسب التشابه
        similar.sort(key=lambda x: x[1], reverse=True)
        return [q for q, _ in similar[:limit]]