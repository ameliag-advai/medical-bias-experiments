#!/usr/bin/env python3
"""
Script to group release test patients by age categories and add age group column.
Age groups: pediatric (0-12), adolescent (13-19), young adult (20-35), 
middle age (36-64), senior (65-100)
"""

import pandas as pd
import sys

def get_age_group(age):
    """Categorize age into predefined groups."""
    if age <= 12:
        return "pediatric"
    elif 13 <= age <= 19:
        return "adolescent"
    elif 20 <= age <= 35:
        return "young_adult"
    elif 36 <= age <= 64:
        return "middle_age"
    elif 65 <= age <= 100:
        return "senior"
    else:
        return "unknown"  # For ages outside expected range

def process_patient_data():
    """Process the release test patients file and add age groups."""
    
    input_file = "/Users/amelia/22406alethia/alethia/release_test_patients"
    output_file = "/Users/amelia/22406alethia/alethia/release-test-patients-age-grouped.csv"
    
    print("📊 Processing release test patients data...")
    
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
        print(f"✅ Loaded {len(df)} patient records")
        
        # Check if AGE column exists
        if 'AGE' not in df.columns:
            print("❌ Error: AGE column not found in the data")
            print(f"Available columns: {list(df.columns)}")
            return False
        
        # Add age group column
        df['age_group'] = df['AGE'].apply(get_age_group)
        
        # Display age group distribution
        print("\n📈 AGE GROUP DISTRIBUTION:")
        age_group_counts = df['age_group'].value_counts().sort_index()
        for group, count in age_group_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {group:12}: {count:4} patients ({percentage:5.1f}%)")
        
        # Display age statistics
        print(f"\n📊 AGE STATISTICS:")
        print(f"  Min age: {df['AGE'].min()}")
        print(f"  Max age: {df['AGE'].max()}")
        print(f"  Mean age: {df['AGE'].mean():.1f}")
        print(f"  Median age: {df['AGE'].median():.1f}")
        
        # Save the updated CSV
        df.to_csv(output_file, index=False)
        print(f"\n✅ Saved grouped data to: {output_file}")
        
        # Show sample of the new data
        print(f"\n📋 SAMPLE OF GROUPED DATA:")
        sample_cols = ['AGE', 'SEX', 'PATHOLOGY', 'age_group']
        if all(col in df.columns for col in sample_cols):
            print(df[sample_cols].head(10).to_string(index=False))
        
        return True
        
    except FileNotFoundError:
        print(f"❌ Error: File not found: {input_file}")
        return False
    except Exception as e:
        print(f"❌ Error processing file: {str(e)}")
        return False

def analyze_age_distribution():
    """Analyze the age distribution in detail."""
    
    input_file = "/Users/amelia/22406alethia/alethia/release_test_patients"
    
    try:
        df = pd.read_csv(input_file)
        
        print("\n🔍 DETAILED AGE ANALYSIS:")
        
        # Age ranges for each group
        age_ranges = {
            'pediatric': (0, 12),
            'adolescent': (13, 19), 
            'young_adult': (20, 35),
            'middle_age': (36, 64),
            'senior': (65, 100)
        }
        
        for group, (min_age, max_age) in age_ranges.items():
            group_data = df[(df['AGE'] >= min_age) & (df['AGE'] <= max_age)]
            if len(group_data) > 0:
                print(f"\n  {group.upper()} ({min_age}-{max_age} years):")
                print(f"    Count: {len(group_data)}")
                print(f"    Age range in data: {group_data['AGE'].min()}-{group_data['AGE'].max()}")
                print(f"    Mean age: {group_data['AGE'].mean():.1f}")
                
                # Sex distribution within age group
                sex_dist = group_data['SEX'].value_counts()
                print(f"    Sex distribution: {dict(sex_dist)}")
        
        # Check for any ages outside expected ranges
        outside_range = df[(df['AGE'] < 0) | (df['AGE'] > 100)]
        if len(outside_range) > 0:
            print(f"\n⚠️  WARNING: {len(outside_range)} patients with ages outside 0-100 range:")
            print(outside_range[['AGE', 'SEX', 'PATHOLOGY']].to_string(index=False))
            
    except Exception as e:
        print(f"❌ Error in analysis: {str(e)}")

def main():
    """Main function to process patient data and add age groups."""
    
    print("🏥 RELEASE TEST PATIENTS - AGE GROUPING")
    print("=" * 50)
    
    # Process the data and add age groups
    success = process_patient_data()
    
    if success:
        # Perform detailed analysis
        analyze_age_distribution()
        
        print("\n" + "=" * 50)
        print("✅ PROCESSING COMPLETE!")
        print(f"📁 Output file: release-test-patients-age-grouped.csv")
        print(f"📊 Age groups: pediatric, adolescent, young_adult, middle_age, senior")
    else:
        print("\n❌ Processing failed. Please check the input file.")
        sys.exit(1)

if __name__ == "__main__":
    main()
