#!/usr/bin/env python3
"""
Direct data fetching script for external sources
"""
import sys
import os

# Add the src directory to the path
sys.path.append('src')

from pages.enhanced_ml import EnhancedAnimeRecommendationSystem

def main():
    print("🚀 Starting external data collection...")
    
    # Initialize the enhanced ML system
    enhanced_system = EnhancedAnimeRecommendationSystem()
    
    print("\n📊 Current cached data status:")
    cached_data = enhanced_system.load_cached_data()
    for source, data in cached_data.items():
        print(f"  {source}: {len(data)} items")
    
    # Fetch AniList data (500 items)
    print("\n🎯 Fetching AniList data (500 items)...")
    try:
        anilist_data = enhanced_system.fetch_anilist_data(500)
        print(f"✅ Successfully fetched {len(anilist_data)} anime from AniList!")
    except Exception as e:
        print(f"❌ Error fetching AniList data: {str(e)}")
    
    # Fetch Kitsu data (500 items - note: Kitsu API may have limits)
    print("\n🐱 Fetching Kitsu data (500 items)...")
    try:
        # Kitsu might have pagination limits, so we'll try 200 first
        kitsu_data = enhanced_system.fetch_kitsu_data(200)
        print(f"✅ Successfully fetched {len(kitsu_data)} anime from Kitsu!")
    except Exception as e:
        print(f"❌ Error fetching Kitsu data: {str(e)}")
    
    # Show final status
    print("\n📈 Final data status:")
    cached_data = enhanced_system.load_cached_data()
    for source, data in cached_data.items():
        print(f"  {source}: {len(data)} items")
    
    print("\n🎉 External data collection complete!")

if __name__ == "__main__":
    main()