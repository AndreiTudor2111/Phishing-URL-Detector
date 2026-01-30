"""
Feature extraction for phishing URL detection.
Combines URL-based, statistical, and pattern-based features.
"""

import re
import numpy as np
from urllib.parse import urlparse
from collections import Counter
from scipy.stats import entropy, ks_2samp
from ..utils.config import Config


def extract_features(url):
    """
    Extract comprehensive features from a URL for phishing detection.
    
    Args:
        url (str): The URL to analyze
        
    Returns:
        dict: Dictionary of extracted features, or None if URL is invalid
    """
    try:
        # Ensure proper URL scheme
        if not url.startswith('http://') and not url.startswith('https://'):
            url = 'http://' + url
        
        parsed_url = urlparse(url)
        hostname = parsed_url.hostname if parsed_url.hostname else ''
        path = parsed_url.path if parsed_url.path else ''
        query = parsed_url.query if parsed_url.query else ''
    
    except (ValueError, Exception) as e:
        # Handle malformed URLs (e.g., invalid IPv6)
        # Return default features with zeros
        return None
    
    # Calculate entropy for a string
    def calculate_entropy(s):
        if len(s) == 0:
            return 0
        probabilities = [float(s.count(c)) / len(s) for c in set(s)]
        return -sum([p * np.log2(p) for p in probabilities if p > 0])
    
    # Calculate character frequencies
    def get_char_freq(s):
        s = s.lower()
        count = Counter(c for c in s if c.isalpha())
        total = sum(count.values())
        if total == 0:
            return [0] * 26
        return [count.get(char, 0) / total for char in 'abcdefghijklmnopqrstuvwxyz']
    
    # Calculate KS statistic
    def calculate_ks_stat(s):
        s_freq = get_char_freq(s)
        eng_freq = [Config.ENGLISH_CHAR_FREQUENCIES[char] / 100 
                   for char in 'abcdefghijklmnopqrstuvwxyz']
        statistic, _ = ks_2samp(s_freq, eng_freq)
        return statistic
    
    # Calculate Kullback-Leibler Divergence
    def kl_divergence(p, q):
        p = np.asarray(p, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        # Add small value to avoid division by zero
        p += 1e-10
        q += 1e-10
        return entropy(p, q)
    
    # Calculate Euclidean distance between frequency distributions
    def euclidean_distance_freq(freq1, freq2):
        return np.linalg.norm(np.array(freq1) - np.array(freq2))
    
    # Check for IP address
    def having_ip_address(host):
        match = re.search(
            '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
            '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'
            '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.'
            '(0x[0-9a-fA-F]{1,2})\\/)|'
            '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}|'
            '[0-9a-fA-F]{7}', host)
        return 1 if match else 0
    
    # Check for shortening service
    def shortening_service(url_str):
        return 1 if re.search(r'bit\\.ly|goo\\.gl|tinyurl|short\\.to|ow\\.ly', url_str) else 0
    
    # Get English character frequencies
    eng_freq = [Config.ENGLISH_CHAR_FREQUENCIES[char] / 100 
               for char in 'abcdefghijklmnopqrstuvwxyz']
    
    # Extract all features
    features = {
        # URL Length and Structure
        'length_url': len(url),
        'length_hostname': len(hostname),
        'length_path': len(path),
        'length_query': len(query),
        'length_tld': len(hostname.split('.')[-1]) if '.' in hostname else 0,
        'avg_token_length': np.mean([len(token) for token in url.split('/') if token]) if any(url.split('/')) else 0,
        
        # Character Counts
        'nb_dots': url.count('.'),
        'nb_hyphens': url.count('-'),
        'nb_slashes': url.count('/'),
        'nb_digits': len(re.findall(r'\\d', url)),
        'nb_underscores': url.count('_'),
        'nb_equals': url.count('='),
        'nb_percent': url.count('%'),
        'nb_questions': url.count('?'),
        'count_at': url.count('@'),
        'count_comma': url.count(','),
        'count_dollar': url.count('$'),
        'count_semicolumn': url.count(';'),
        'count_space': url.count(' '),
        'count_and': url.count('&'),
        'count_double_slash': url.count('//') - 1,  # Minus 1 for http://
        'count_colon': url.count(':') - 1,  # Minus 1 for http:
        'count_star': url.count('*'),
        'count_or': url.count('|'),
        'count_tilde': url.count('~'),
        'count_special_chars': sum(1 for c in url if c in ['@', '!', '$', '%', '^', '&', '*', '(', ')']),
        'number_punctuations': len(re.findall(r'[.!#$%&,.;\']', url)),
        
        # Ratios
        'ratio_digits_url': len(re.findall(r'\\d', url)) / (len(url) + 1e-6),
        'ratio_digits_host': len(re.findall(r'\\d', hostname)) / (len(hostname) + 1e-6),
        'ratio_digits_path': len(re.findall(r'\\d', path)) / (len(path) + 1e-6),
        'ratio_uppercase_url': sum(1 for c in url if c.isupper()) / (len(url) + 1e-6),
        'ratio_length_url_path': len(url) / (len(path) + 1e-6),
        
        # Suspicious Patterns
        'contains_ip': 1 if re.search(r'\\d+\\.\\d+\\.\\d+\\.\\d+', hostname) else 0,
        'having_ip_address': having_ip_address(hostname),
        'is_ip_address': 1 if re.match(r'^https?://\\d+\\.\\d+\\.\\d+\\.\\d+', url) else 0,
        'check_www': 1 if 'www' in hostname else 0,
        'check_com': 1 if '.com' in hostname else 0,
        'shortening_service': shortening_service(url),
        'suspicious_keywords': 1 if any(word in url.lower() for word in Config.SUSPICIOUS_WORDS) else 0,
        'number_suspicious_words': sum(word in url.lower() for word in Config.SUSPICIOUS_WORDS),
        'phish_hints': sum(path.lower().count(hint) for hint in Config.PHISHING_HINTS),
        
        # Subdomains
        'count_subdomains': hostname.count('.') - 1 if hostname else 0,
        'abnormal_subdomain': 1 if re.search(r'(http[s]?://(w[w]?|\\d))([w]?(\\d|-))', url) else 0,
        
        # Query Parameters
        'nb_queries': query.count('&') + 1 if query else 0,
        'has_multiple_queries': 1 if query.count('&') > 2 else 0,
        
        # File and Path Indicators
        'path_extension': 1 if re.search(r'\\.(exe|zip|pdf|js|html|php|asp|key|pem|crt|conf|db)$', path) else 0,
        'is_sensitive_file': 1 if re.search(r'\\.(key|pem|crt|conf|db)$', path) else 0,
        'number_additional_tlds': len(re.findall(r'\\.[a-z]{2,}', path)) - 1 if len(re.findall(r'\\.[a-z]{2,}', path)) > 0 else 0,
        
        # TLD Features
        'tld_in_path': 1 if hostname and hostname.split('.')[-1] in path else 0,
        'suspecious_tld': 1 if hostname and hostname.split('.')[-1] in Config.SUSPICIOUS_TLDS else 0,
        
        # Protocol Features
        'count_http_token': path.count('http'),
        'https_token': 1 if parsed_url.scheme == 'https' else 0,
        'count_http': url.lower().count('http'),
        'count_redirection': url.count('//') - 1,
        
        # Entropy
        'entropy_url': calculate_entropy(url),
        
        # Words and Tokens
        'avg_word_length': np.mean([len(word) for word in re.findall(r'\\w+', hostname)]) if re.findall(r'\\w+', hostname) else 0,
        'total_words': len(re.findall(r'\\w+', hostname)),
        
        # Statistical Measures
        'euclidean_distance': euclidean_distance_freq(get_char_freq(url), eng_freq),
        'ks_statistic': calculate_ks_stat(url),
        'kl_divergence': kl_divergence(get_char_freq(url), eng_freq),
    }
    
    return features


def extract_features_batch(urls):
    """
    Extract features from multiple URLs.
    
    Args:
        urls (list): List of URLs to process
        
    Returns:
        list: List of feature dictionaries
    """
    return [extract_features(url) for url in urls]
