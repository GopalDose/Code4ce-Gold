import React, { useContext } from 'react';
import './UpperHeader.css';
import { AuthContext } from '../../context/AuthContext'; // Import the AuthContext

const languageOptions = [
  { code: 'ar', name: 'Arabic' },
  { code: 'de', name: 'German' },
  { code: 'en', name: 'English' },
  { code: 'es', name: 'Spanish' },
  { code: 'fr', name: 'French' },
  { code: 'he', name: 'Hebrew' },
  { code: 'it', name: 'Italian' },
  { code: 'nl', name: 'Dutch' },
  { code: 'no', name: 'Norwegian' },
  { code: 'pt', name: 'Portuguese' },
  { code: 'ru', name: 'Russian' },
  { code: 'sv', name: 'Swedish' },
  { code: 'ud', name: 'Urdu' },
  { code: 'zh', name: 'Chinese' }
];

const UpperHeader = () => {
  const { language, setLanguage } = useContext(AuthContext); // Access language and setLanguage from context

  const handleLanguageChange = (event) => {
    setLanguage(event.target.value); // Update language in context when user selects a new language
  };

  return (
    <div className="upperheader">
      <div className="language">
        <label htmlFor="language-select">Language: </label>
        <select
          name="language"
          id="language-select"
          value={language}
          onChange={handleLanguageChange} // Handle language change
        >
          {languageOptions.map((lang) => (
            <option key={lang.code} value={lang.code}>
              {lang.name} ({lang.code})
            </option>
          ))}
        </select>
      </div>
    </div>
  );
};

export default UpperHeader;
