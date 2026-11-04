from supabase_client import supabase
def extract_movie_features_by_ids(movie_ids : list[int]):
    res = (
    supabase.table("movies")
    .select("id, title, features")
    .in_("id", movie_ids)
    .execute()
    )
    return res